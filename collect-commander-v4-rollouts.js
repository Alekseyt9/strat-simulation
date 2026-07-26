import { mkdir, writeFile } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import { dirname, resolve } from 'node:path';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import { Battle } from './diagnostic.js';
import { createV4Policy, isCompatibleV4Policy } from './commander-v4-policy.js';
import { TRAINED_COMMANDER_V4_POLICY } from './trained-commander-v4-policy.js';

function parseArgs(argv) {
  const options = {
    battles: 192,
    workers: availableParallelism(),
    seed: 490019,
    exploration: 0.12,
    output: '.training/commander-v4-rollouts.json',
    evaluate: false,
    opponents: ['crowd'],
    forcedDoctrine: null,
    forcedFocus: null,
    forcedFire: null
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--battles') options.battles = Number(argv[++index]);
    else if (arg === '--workers') options.workers = Number(argv[++index]);
    else if (arg === '--seed') options.seed = Number(argv[++index]);
    else if (arg === '--exploration') options.exploration = Number(argv[++index]);
    else if (arg === '--output') options.output = argv[++index];
    else if (arg === '--evaluate') options.evaluate = true;
    else if (arg === '--opponents') options.opponents = argv[++index].split(',').filter(Boolean);
    else if (arg === '--forced-doctrine') options.forcedDoctrine = Number(argv[++index]);
    else if (arg === '--forced-focus') options.forcedFocus = Number(argv[++index]);
    else if (arg === '--forced-fire') options.forcedFire = Number(argv[++index]);
  }
  return options;
}

function makeRng(initialSeed) {
  let state = initialSeed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function makeScenarios(count, seed, opponents) {
  const rand = makeRng(seed);
  const unitOptions = [80, 100, 140, 180, 240, 320, 400];
  const scenarios = [];
  for (let pair = 0; scenarios.length < count; pair++) {
    const battleSeed = ((seed + pair * 7919 + rand() * 1e8) | 0) >>> 0;
    const units = unitOptions[(rand() * unitOptions.length) | 0];
    const opponent = opponents[pair % opponents.length];
    for (let v4Team = 0; v4Team < 2 && scenarios.length < count; v4Team++) {
      scenarios.push({ id: scenarios.length, pair, opponent, units, v4Team, seed: battleSeed });
    }
  }
  return scenarios;
}

function collectChunk({
  scenarios,
  policy,
  evaluate,
  exploration,
  forcedDoctrine,
  forcedFocus,
  forcedFire
}) {
  const samples = [];
  const summary = {
    battles: scenarios.length, wins: 0, losses: 0, draws: 0,
    duration: 0, timedOut: 0, actions: {}, byOpponent: {}
  };
  for (const scenario of scenarios) {
    const blue = scenario.v4Team === 0 ? 'commander_v4' : scenario.opponent;
    const red = scenario.v4Team === 1 ? 'commander_v4' : scenario.opponent;
    const policies = scenario.v4Team === 0 ? [policy, null] : [null, policy];
    const recorder = {
      record: !evaluate,
      sample: !evaluate,
      exploration,
      forcedDoctrine,
      forcedFocus,
      forcedFire
    };
    const usesForcedPlan = Number.isInteger(forcedDoctrine)
      || Number.isInteger(forcedFocus)
      || Number.isInteger(forcedFire);
    const training = evaluate && !usesForcedPlan
      ? [null, null]
      : scenario.v4Team === 0 ? [recorder, null] : [null, recorder];
    const result = new Battle({
      blue, red, units: scenario.units, seed: scenario.seed, policies, training
    }).run();
    const expected = scenario.v4Team === 0 ? 'blue' : 'red';
    const bucket = summary.byOpponent[scenario.opponent] ??= {
      wins: 0, losses: 0, draws: 0, duration: 0, timedOut: 0
    };
    if (result.winner === expected) summary.wins++, bucket.wins++;
    else if (result.winner === 'draw') summary.draws++, bucket.draws++;
    else summary.losses++, bucket.losses++;
    summary.duration += result.duration;
    bucket.duration += result.duration;
    if (result.duration >= 239.9) summary.timedOut++, bucket.timedOut++;
    for (const [name, count] of Object.entries(result.commanderDecisions[scenario.v4Team])) {
      summary.actions[name] = (summary.actions[name] ?? 0) + count;
    }
    if (!evaluate) {
      const trajectory = result.trainingSamples[scenario.v4Team];
      const episodeWeight = 1 / Math.max(1, trajectory.length);
      samples.push(...trajectory.map((sample, step) => ({
        ...sample, episode: scenario.id, pair: scenario.pair, step, episodeWeight
      })));
    }
  }
  return { samples, summary };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const policy = isCompatibleV4Policy(TRAINED_COMMANDER_V4_POLICY)
    ? TRAINED_COMMANDER_V4_POLICY
    : createV4Policy();
  const scenarios = makeScenarios(options.battles, options.seed, options.opponents);
  const workerCount = Math.max(1, Math.min(options.workers, options.battles));
  const chunks = Array.from({ length: workerCount }, () => []);
  scenarios.forEach((scenario, index) => chunks[index % workerCount].push(scenario));
  const parts = await Promise.all(chunks.map(chunk => new Promise((done, fail) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: {
        v4Worker: true,
        scenarios: chunk,
        policy,
        evaluate: options.evaluate,
        exploration: options.exploration,
        forcedDoctrine: options.forcedDoctrine,
        forcedFocus: options.forcedFocus,
        forcedFire: options.forcedFire
      }
    });
    worker.once('message', done);
    worker.once('error', fail);
  })));
  const summary = parts.reduce((total, part) => {
    for (const field of ['battles', 'wins', 'losses', 'draws', 'duration', 'timedOut']) {
      total[field] += part.summary[field];
    }
    for (const [name, count] of Object.entries(part.summary.actions)) {
      total.actions[name] = (total.actions[name] ?? 0) + count;
    }
    for (const [name, values] of Object.entries(part.summary.byOpponent)) {
      const bucket = total.byOpponent[name] ??= {
        wins: 0, losses: 0, draws: 0, duration: 0, timedOut: 0
      };
      for (const field of ['wins', 'losses', 'draws', 'duration', 'timedOut']) {
        bucket[field] += values[field];
      }
    }
    return total;
  }, {
    battles: 0, wins: 0, losses: 0, draws: 0, duration: 0, timedOut: 0,
    actions: {}, byOpponent: {}
  });
  summary.averageDuration = summary.duration / Math.max(1, summary.battles);
  const outputPath = resolve(options.output);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, JSON.stringify({
    policy, options, summary, samples: options.evaluate ? [] : parts.flatMap(part => part.samples)
  }), 'utf8');
  console.log(JSON.stringify(summary));
}

if (isMainThread) await main();
else if (workerData?.v4Worker) parentPort.postMessage(collectChunk(workerData));
