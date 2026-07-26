import { mkdir, writeFile } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import { dirname, resolve } from 'node:path';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import { Battle } from './diagnostic.js';
import { createPolicy, isCompatiblePolicy } from './neural-policy.js';
import { TRAINED_POLICY } from './trained-policy.js';

function parseArgs(argv) {
  const options = {
    battles: 160,
    workers: availableParallelism(),
    seed: 88031,
    temperature: 0.32,
    epsilon: 0.16,
    output: '.training/rollouts.json',
    evaluate: false,
    opponents: null
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--battles') options.battles = Number(argv[++i]);
    else if (arg === '--workers') options.workers = Number(argv[++i]);
    else if (arg === '--seed') options.seed = Number(argv[++i]);
    else if (arg === '--temperature') options.temperature = Number(argv[++i]);
    else if (arg === '--epsilon') options.epsilon = Number(argv[++i]);
    else if (arg === '--output') options.output = argv[++i];
    else if (arg === '--evaluate') options.evaluate = true;
    else if (arg === '--opponents') options.opponents = argv[++i].split(',').filter(Boolean);
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

function makeScenarios(count, seed, requestedOpponents = null) {
  const rand = makeRng(seed);
  const unitOptions = [80, 100, 140, 180, 240, 320, 400];
  const opponents = requestedOpponents?.length
    ? requestedOpponents
    : ['crowd', 'crowd', 'crowd', 'crowd', 'crowd', 'adaptive'];
  return Array.from({ length: count }, (_, index) => ({
    opponent: opponents[index % opponents.length],
    units: unitOptions[(rand() * unitOptions.length) | 0],
    neuralTeam: index % 2,
    seed: ((seed + index * 7919 + rand() * 1e8) | 0) >>> 0
  }));
}

function collectChunk({ scenarios, policy, temperature, epsilon, evaluate }) {
  const samples = [];
  const summary = {
    wins: 0, losses: 0, draws: 0, battles: scenarios.length, byOpponent: {}
  };
  for (const scenario of scenarios) {
    const blue = scenario.neuralTeam === 0 ? 'neural' : scenario.opponent;
    const red = scenario.neuralTeam === 1 ? 'neural' : scenario.opponent;
    const policies = scenario.neuralTeam === 0 ? [policy, null] : [null, policy];
    const training = evaluate
      ? [null, null]
      : scenario.neuralTeam === 0
        ? [{ record: true, temperature, epsilon }, null]
        : [null, { record: true, temperature, epsilon }];
    const battle = new Battle({
      blue,
      red,
      units: scenario.units,
      seed: scenario.seed,
      policies,
      training
    });
    const result = battle.run();
    const neuralWinner = scenario.neuralTeam === 0 ? 'blue' : 'red';
    const bucket = summary.byOpponent[scenario.opponent] ??= {
      wins: 0, losses: 0, draws: 0
    };
    if (result.winner === neuralWinner) summary.wins++, bucket.wins++;
    else if (result.winner === 'draw') summary.draws++, bucket.draws++;
    else summary.losses++, bucket.losses++;
    if (!evaluate) samples.push(...result.trainingSamples[scenario.neuralTeam]);
  }
  return { samples, summary };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const policy = isCompatiblePolicy(TRAINED_POLICY) ? TRAINED_POLICY : createPolicy();
  const scenarios = makeScenarios(options.battles, options.seed, options.opponents);
  const workerCount = Math.max(1, Math.min(options.workers, options.battles));
  const chunks = Array.from({ length: workerCount }, () => []);
  scenarios.forEach((scenario, index) => chunks[index % workerCount].push(scenario));
  const workers = chunks.map(chunk => new Promise((resolve, reject) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: {
        rolloutWorker: true,
        scenarios: chunk,
        policy,
        temperature: options.temperature,
        epsilon: options.epsilon,
        evaluate: options.evaluate
      }
    });
    worker.once('message', resolve);
    worker.once('error', reject);
  }));
  const parts = await Promise.all(workers);
  const samples = options.evaluate ? [] : parts.flatMap(part => part.samples);
  const summary = parts.reduce((total, part) => {
    total.battles += part.summary.battles;
    total.wins += part.summary.wins;
    total.losses += part.summary.losses;
    total.draws += part.summary.draws;
    for (const [name, values] of Object.entries(part.summary.byOpponent)) {
      const bucket = total.byOpponent[name] ??= { wins: 0, losses: 0, draws: 0 };
      bucket.wins += values.wins;
      bucket.losses += values.losses;
      bucket.draws += values.draws;
    }
    return total;
  }, { battles: 0, wins: 0, losses: 0, draws: 0, byOpponent: {} });
  const outputPath = resolve(options.output);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, JSON.stringify({ policy, options, summary, samples }), 'utf8');
  console.log(`Траектории: ${summary.wins}-${summary.losses}-${summary.draws}, ${samples.length} решений → ${options.output}`);
}

if (isMainThread) await main();
else if (workerData?.rolloutWorker) parentPort.postMessage(collectChunk(workerData));
