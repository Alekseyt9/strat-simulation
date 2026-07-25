import { mkdir, writeFile } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import { dirname, resolve } from 'node:path';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import { Battle } from './diagnostic.js';
import { createV3Policy, isCompatibleV3Policy } from './commander-v3-policy.js';
import { TRAINED_COMMANDER_V3_POLICY } from './trained-commander-v3-policy.js';

function parseArgs(argv) {
  const options = {
    battles: 192,
    workers: availableParallelism(),
    seed: 371903,
    exploration: 0.12,
    output: '.training/commander-v3-rollouts.json',
    evaluate: false,
    opponents: ['crowd']
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
    for (let v3Team = 0; v3Team < 2 && scenarios.length < count; v3Team++) {
      scenarios.push({
        id: scenarios.length,
        pair,
        opponent,
        units,
        v3Team,
        seed: battleSeed
      });
    }
  }
  return scenarios;
}

function collectChunk({ scenarios, policy, evaluate, exploration }) {
  const samples = [];
  const summary = {
    battles: scenarios.length,
    wins: 0,
    losses: 0,
    draws: 0,
    duration: 0,
    timedOut: 0,
    actions: { hold: 0, advance: 0, assault: 0, reserve: 0, flank: 0 },
    byOpponent: {}
  };
  for (const scenario of scenarios) {
    const blue = scenario.v3Team === 0 ? 'commander_v3' : scenario.opponent;
    const red = scenario.v3Team === 1 ? 'commander_v3' : scenario.opponent;
    const policies = scenario.v3Team === 0 ? [policy, null] : [null, policy];
    const training = evaluate
      ? [null, null]
      : scenario.v3Team === 0
        ? [{ record: true, sample: true, exploration }, null]
        : [null, { record: true, sample: true, exploration }];
    const result = new Battle({
      blue,
      red,
      units: scenario.units,
      seed: scenario.seed,
      policies,
      training
    }).run();
    const expected = scenario.v3Team === 0 ? 'blue' : 'red';
    const bucket = summary.byOpponent[scenario.opponent] ??= {
      wins: 0, losses: 0, draws: 0, duration: 0, timedOut: 0
    };
    if (result.winner === expected) summary.wins++, bucket.wins++;
    else if (result.winner === 'draw') summary.draws++, bucket.draws++;
    else summary.losses++, bucket.losses++;
    summary.duration += result.duration;
    bucket.duration += result.duration;
    if (result.duration >= 239.9) summary.timedOut++, bucket.timedOut++;
    for (const [name, count] of Object.entries(result.commanderDecisions[scenario.v3Team])) {
      summary.actions[name] = (summary.actions[name] ?? 0) + count;
    }
    if (!evaluate) {
      const trajectory = result.trainingSamples[scenario.v3Team];
      const episodeWeight = 1 / Math.max(1, trajectory.length);
      samples.push(...trajectory.map((sample, step) => ({
        ...sample,
        episode: scenario.id,
        pair: scenario.pair,
        step,
        episodeWeight
      })));
    }
  }
  return { samples, summary };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const policy = isCompatibleV3Policy(TRAINED_COMMANDER_V3_POLICY)
    ? TRAINED_COMMANDER_V3_POLICY
    : createV3Policy();
  const scenarios = makeScenarios(options.battles, options.seed, options.opponents);
  const workerCount = Math.max(1, Math.min(options.workers, options.battles));
  const chunks = Array.from({ length: workerCount }, () => []);
  scenarios.forEach((scenario, index) => chunks[index % workerCount].push(scenario));
  const parts = await Promise.all(chunks.map(chunk => new Promise((resolvePart, rejectPart) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: {
        commanderV3Worker: true,
        scenarios: chunk,
        policy,
        evaluate: options.evaluate,
        exploration: options.exploration
      }
    });
    worker.once('message', resolvePart);
    worker.once('error', rejectPart);
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
  const samples = options.evaluate ? [] : parts.flatMap(part => part.samples);
  const outputPath = resolve(options.output);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, JSON.stringify({ policy, options, summary, samples }), 'utf8');
  console.log(JSON.stringify(summary));
}

if (isMainThread) await main();
else if (workerData?.commanderV3Worker) {
  parentPort.postMessage(collectChunk(workerData));
}
