import { availableParallelism } from 'node:os';
import { writeFile } from 'node:fs/promises';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import { Battle } from './diagnostic.js';
import {
  createPolicy,
  isCompatiblePolicy,
  parameterCount,
  POLICY_STANCES
} from './neural-policy.js';
import { TRAINED_POLICY } from './trained-policy.js';

function parseArgs(argv) {
  const options = {
    generations: 10,
    population: 12,
    battles: 12,
    workers: availableParallelism(),
    seed: 74021,
    sigma: 0.16
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--generations') options.generations = Number(argv[++i]);
    else if (arg === '--population') options.population = Number(argv[++i]);
    else if (arg === '--battles') options.battles = Number(argv[++i]);
    else if (arg === '--workers') options.workers = Number(argv[++i]);
    else if (arg === '--seed') options.seed = Number(argv[++i]);
    else if (arg === '--sigma') options.sigma = Number(argv[++i]);
  }
  for (const key of ['generations', 'population', 'battles', 'workers']) {
    if (!Number.isFinite(options[key]) || options[key] < 1) throw new Error(`Неверный --${key}`);
    options[key] = Math.floor(options[key]);
  }
  if (!Number.isFinite(options.sigma) || options.sigma <= 0) throw new Error('Неверный --sigma');
  return options;
}

function makeRng(initialSeed) {
  let state = initialSeed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function gaussian(rand) {
  const u = Math.max(1e-9, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}

function mutate(policy, sigma, rand) {
  return {
    ...policy,
    weights: policy.weights.map(weight => weight + gaussian(rand) * sigma)
  };
}

function makeScenarios(count, seed, holdout = false) {
  const rand = makeRng(seed);
  const units = holdout ? [90, 150, 220, 320, 450] : [80, 120, 180, 260, 400];
  const opponents = ['crowd', 'crowd', 'crowd', 'offensive', 'defensive', 'adaptive'];
  return Array.from({ length: count }, (_, index) => ({
    opponent: opponents[index % opponents.length],
    units: units[(rand() * units.length) | 0],
    neuralTeam: index % 2,
    seed: ((seed + index * 7919 + rand() * 1e7) | 0) >>> 0
  }));
}

function evaluate(policy, scenarios) {
  let score = 0;
  let wins = 0;
  let losses = 0;
  let draws = 0;
  let survivorEdge = 0;
  const actions = Object.fromEntries(POLICY_STANCES.map(action => [action, 0]));
  for (const scenario of scenarios) {
    const blue = scenario.neuralTeam === 0 ? 'neural' : scenario.opponent;
    const red = scenario.neuralTeam === 1 ? 'neural' : scenario.opponent;
    const policies = scenario.neuralTeam === 0 ? [policy, null] : [null, policy];
    const battle = new Battle({ blue, red, units: scenario.units, seed: scenario.seed, policies });
    const result = battle.run();
    const neuralWon = result.winner === (scenario.neuralTeam === 0 ? 'blue' : 'red');
    const neuralLost = result.winner === (scenario.neuralTeam === 0 ? 'red' : 'blue');
    if (neuralWon) wins++;
    else if (neuralLost) losses++;
    else draws++;
    const own = result.survivors[scenario.neuralTeam];
    const enemy = result.survivors[1 - scenario.neuralTeam];
    const edge = (own - enemy) / scenario.units;
    survivorEdge += edge;
    score += (neuralWon ? 1 : neuralLost ? -1 : 0) + edge * 0.45;
    const decisions = result.commanderDecisions[scenario.neuralTeam];
    for (const action of Object.keys(actions)) actions[action] += decisions[action] ?? 0;
  }
  return {
    score: score / scenarios.length,
    wins,
    losses,
    draws,
    survivorEdge: survivorEdge / scenarios.length,
    actions
  };
}

class TrainingPool {
  constructor(size) {
    this.queue = [];
    this.slots = Array.from({ length: size }, () => {
      const worker = new Worker(new URL(import.meta.url), { workerData: { trainerWorker: true } });
      const slot = { worker, current: null };
      worker.on('message', result => {
        const current = slot.current;
        slot.current = null;
        current.resolve(result);
        this.dispatch(slot);
      });
      worker.on('error', error => {
        const current = slot.current;
        slot.current = null;
        current?.reject(error);
        this.dispatch(slot);
      });
      return slot;
    });
  }

  run(job) {
    return new Promise((resolve, reject) => {
      this.queue.push({ job, resolve, reject });
      for (const slot of this.slots) this.dispatch(slot);
    });
  }

  dispatch(slot) {
    if (slot.current || !this.queue.length) return;
    slot.current = this.queue.shift();
    slot.worker.postMessage(slot.current.job);
  }

  async close() {
    await Promise.all(this.slots.map(slot => slot.worker.terminate()));
  }
}

async function savePolicy(policy, metadata) {
  const source = `// Generated locally by train-commander.js.\n`
    + `export const TRAINED_POLICY = ${JSON.stringify(policy)};\n`
    + `export const TRAINING_METADATA = ${JSON.stringify(metadata, null, 2)};\n`;
  await writeFile(new URL('./trained-policy.js', import.meta.url), source, 'utf8');
}

async function train() {
  const options = parseArgs(process.argv.slice(2));
  const rand = makeRng(options.seed);
  const poolSize = Math.max(1, Math.min(options.workers, options.population));
  const pool = new TrainingPool(poolSize);
  let champion = isCompatiblePolicy(TRAINED_POLICY)
    ? structuredClone(TRAINED_POLICY)
    : createPolicy();
  let bestEver = { policy: champion, score: -Infinity, generation: -1 };
  const fixedValidation = makeScenarios(
    Math.max(8, Math.ceil(options.battles * 0.6)),
    options.seed + 555557,
    true
  );
  const started = performance.now();

  try {
    for (let generation = 0; generation < options.generations; generation++) {
      const sigma = options.sigma * Math.pow(0.88, generation);
      const scenarios = makeScenarios(options.battles, options.seed + generation * 104729);
      const population = [champion];
      while (population.length < options.population) population.push(mutate(champion, sigma, rand));
      const results = await Promise.all(population.map(policy => pool.run({ policy, scenarios })));
      const ranked = results
        .map((result, index) => ({ ...result, policy: population[index], index }))
        .sort((a, b) => b.score - a.score);
      const finalists = ranked.slice(0, Math.min(3, ranked.length));
      const validationResults = await Promise.all(
        finalists.map(candidate => pool.run({ policy: candidate.policy, scenarios: fixedValidation }))
      );
      for (let index = 0; index < finalists.length; index++) {
        finalists[index].validation = validationResults[index];
        finalists[index].selectionScore = finalists[index].score * 0.6
          + validationResults[index].score * 0.4;
      }
      finalists.sort((a, b) => b.selectionScore - a.selectionScore);
      const selected = finalists[0];
      champion = structuredClone(selected.policy);
      if (selected.validation.score > bestEver.score) {
        bestEver = { policy: selected.policy, score: selected.validation.score, generation };
      }
      console.log(
        `Поколение ${generation + 1}/${options.generations}: `
        + `train=${selected.score.toFixed(3)}, validation=${selected.validation.score.toFixed(3)}, `
        + `${selected.wins}-${selected.losses}-${selected.draws}, `
        + `остаток=${(selected.survivorEdge * 100).toFixed(1)}%, `
        + `действия=${JSON.stringify(selected.actions)}`
      );
    }
  } finally {
    await pool.close();
  }

  const holdout = makeScenarios(Math.max(30, options.battles * 3), options.seed + 999983, true);
  const holdoutResult = evaluate(bestEver.policy, holdout);
  const metadata = {
    trainedAt: new Date().toISOString(),
    options,
    parameterCount: parameterCount(),
    bestGeneration: bestEver.generation + 1,
    trainingScore: bestEver.score,
    holdout: holdoutResult,
    elapsedSeconds: (performance.now() - started) / 1000
  };
  await savePolicy(bestEver.policy, metadata);
  console.log(`Контроль: ${holdoutResult.wins}-${holdoutResult.losses}-${holdoutResult.draws}, score=${holdoutResult.score.toFixed(3)}`);
  console.log(`Модель сохранена в trained-policy.js за ${metadata.elapsedSeconds.toFixed(1)} с`);
}

if (isMainThread) {
  await train();
} else if (workerData?.trainerWorker) {
  parentPort.on('message', ({ policy, scenarios }) => {
    parentPort.postMessage(evaluate(policy, scenarios));
  });
}
