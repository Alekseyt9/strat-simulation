import { mkdir, writeFile } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import { dirname, resolve } from 'node:path';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import { Battle } from './diagnostic.js';
import { createPpoPolicy, isCompatiblePpoPolicy } from './ppo-policy.js';
import { TRAINED_PPO_POLICY } from './trained-ppo-policy.js';

function parseArgs(argv) {
  const options = {
    battles: 192,
    workers: availableParallelism(),
    seed: 91571,
    output: '.training/ppo-rollouts.json',
    evaluate: false,
    opponents: null
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--battles') options.battles = Number(argv[++index]);
    else if (arg === '--workers') options.workers = Number(argv[++index]);
    else if (arg === '--seed') options.seed = Number(argv[++index]);
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

function makeScenarios(count, seed, evaluate, requestedOpponents = null) {
  const rand = makeRng(seed);
  const units = [80, 100, 140, 180, 240, 320, 400];
  const trainingOpponents = ['crowd', 'crowd', 'crowd', 'adaptive', 'offensive', 'defensive'];
  const evaluationOpponents = ['crowd', 'adaptive', 'offensive', 'defensive'];
  const opponents = requestedOpponents?.length
    ? requestedOpponents
    : evaluate ? evaluationOpponents : trainingOpponents;
  return Array.from({ length: count }, (_, index) => ({
    id: index,
    opponent: opponents[index % opponents.length],
    units: units[(rand() * units.length) | 0],
    ppoTeam: index % 2,
    seed: ((seed + index * 7919 + rand() * 1e8) | 0) >>> 0
  }));
}

function collectChunk({ scenarios, policy, evaluate }) {
  const samples = [];
  const summary = {
    battles: scenarios.length,
    wins: 0,
    losses: 0,
    draws: 0,
    duration: 0,
    timedOut: 0,
    byOpponent: {}
  };
  for (const scenario of scenarios) {
    const blue = scenario.ppoTeam === 0 ? 'ppo' : scenario.opponent;
    const red = scenario.ppoTeam === 1 ? 'ppo' : scenario.opponent;
    const policies = scenario.ppoTeam === 0 ? [policy, null] : [null, policy];
    const training = evaluate
      ? [null, null]
      : scenario.ppoTeam === 0
        ? [{ record: true, sample: true }, null]
        : [null, { record: true, sample: true }];
    const result = new Battle({
      blue,
      red,
      units: scenario.units,
      seed: scenario.seed,
      policies,
      training
    }).run();
    const expectedWinner = scenario.ppoTeam === 0 ? 'blue' : 'red';
    summary.duration += result.duration;
    if (result.duration >= 239.9) summary.timedOut++;
    const bucket = summary.byOpponent[scenario.opponent] ??= {
      wins: 0, losses: 0, draws: 0, duration: 0, timedOut: 0
    };
    bucket.duration += result.duration;
    if (result.duration >= 239.9) bucket.timedOut++;
    if (result.winner === expectedWinner) summary.wins++, bucket.wins++;
    else if (result.winner === 'draw') summary.draws++, bucket.draws++;
    else summary.losses++, bucket.losses++;
    if (!evaluate) {
      samples.push(...result.trainingSamples[scenario.ppoTeam].map(sample => ({
        ...sample,
        episode: scenario.id
      })));
    }
  }
  return { samples, summary };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const policy = isCompatiblePpoPolicy(TRAINED_PPO_POLICY)
    ? TRAINED_PPO_POLICY
    : createPpoPolicy();
  const scenarios = makeScenarios(options.battles, options.seed, options.evaluate, options.opponents);
  const workerCount = Math.max(1, Math.min(options.workers, options.battles));
  const chunks = Array.from({ length: workerCount }, () => []);
  scenarios.forEach((scenario, index) => chunks[index % workerCount].push(scenario));
  const parts = await Promise.all(chunks.map(chunk => new Promise((resolvePart, rejectPart) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: { ppoRolloutWorker: true, scenarios: chunk, policy, evaluate: options.evaluate }
    });
    worker.once('message', resolvePart);
    worker.once('error', rejectPart);
  })));
  const summary = parts.reduce((total, part) => {
    total.battles += part.summary.battles;
    total.wins += part.summary.wins;
    total.losses += part.summary.losses;
    total.draws += part.summary.draws;
    total.duration += part.summary.duration;
    total.timedOut += part.summary.timedOut;
    for (const [name, values] of Object.entries(part.summary.byOpponent)) {
      const bucket = total.byOpponent[name] ??= {
        wins: 0, losses: 0, draws: 0, duration: 0, timedOut: 0
      };
      bucket.wins += values.wins;
      bucket.losses += values.losses;
      bucket.draws += values.draws;
      bucket.duration += values.duration;
      bucket.timedOut += values.timedOut;
    }
    return total;
  }, { battles: 0, wins: 0, losses: 0, draws: 0, duration: 0, timedOut: 0, byOpponent: {} });
  summary.averageDuration = summary.duration / Math.max(1, summary.battles);
  const samples = options.evaluate ? [] : parts.flatMap(part => part.samples);
  const outputPath = resolve(options.output);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, JSON.stringify({ policy, options, summary, samples }), 'utf8');
  console.log(JSON.stringify(summary));
}

if (isMainThread) await main();
else if (workerData?.ppoRolloutWorker) {
  parentPort.postMessage(collectChunk(workerData));
}
