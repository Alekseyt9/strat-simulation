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
    output: '.training/rollouts.json'
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--battles') options.battles = Number(argv[++i]);
    else if (arg === '--workers') options.workers = Number(argv[++i]);
    else if (arg === '--seed') options.seed = Number(argv[++i]);
    else if (arg === '--temperature') options.temperature = Number(argv[++i]);
    else if (arg === '--epsilon') options.epsilon = Number(argv[++i]);
    else if (arg === '--output') options.output = argv[++i];
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

function makeScenarios(count, seed) {
  const rand = makeRng(seed);
  const unitOptions = [80, 100, 140, 180, 240, 320, 400];
  const opponents = ['crowd', 'crowd', 'crowd', 'crowd', 'crowd', 'adaptive'];
  return Array.from({ length: count }, (_, index) => ({
    opponent: opponents[index % opponents.length],
    units: unitOptions[(rand() * unitOptions.length) | 0],
    neuralTeam: index % 2,
    seed: ((seed + index * 7919 + rand() * 1e8) | 0) >>> 0
  }));
}

function collectChunk({ scenarios, policy, temperature, epsilon }) {
  const samples = [];
  const summary = { wins: 0, losses: 0, draws: 0, battles: scenarios.length };
  for (const scenario of scenarios) {
    const blue = scenario.neuralTeam === 0 ? 'neural' : scenario.opponent;
    const red = scenario.neuralTeam === 1 ? 'neural' : scenario.opponent;
    const policies = scenario.neuralTeam === 0 ? [policy, null] : [null, policy];
    const training = scenario.neuralTeam === 0
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
    if (result.winner === neuralWinner) summary.wins++;
    else if (result.winner === 'draw') summary.draws++;
    else summary.losses++;
    samples.push(...result.trainingSamples[scenario.neuralTeam]);
  }
  return { samples, summary };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const policy = isCompatiblePolicy(TRAINED_POLICY) ? TRAINED_POLICY : createPolicy();
  const scenarios = makeScenarios(options.battles, options.seed);
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
        epsilon: options.epsilon
      }
    });
    worker.once('message', resolve);
    worker.once('error', reject);
  }));
  const parts = await Promise.all(workers);
  const samples = parts.flatMap(part => part.samples);
  const summary = parts.reduce((total, part) => ({
    battles: total.battles + part.summary.battles,
    wins: total.wins + part.summary.wins,
    losses: total.losses + part.summary.losses,
    draws: total.draws + part.summary.draws
  }), { battles: 0, wins: 0, losses: 0, draws: 0 });
  const outputPath = resolve(options.output);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, JSON.stringify({ policy, options, summary, samples }), 'utf8');
  console.log(`Траектории: ${summary.wins}-${summary.losses}-${summary.draws}, ${samples.length} решений → ${options.output}`);
}

if (isMainThread) await main();
else if (workerData?.rolloutWorker) parentPort.postMessage(collectChunk(workerData));
