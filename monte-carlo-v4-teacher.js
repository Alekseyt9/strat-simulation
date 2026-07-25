import { mkdir, writeFile } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import { dirname, resolve } from 'node:path';
import { isMainThread, parentPort, Worker, workerData } from 'node:worker_threads';
import { Battle } from './diagnostic.js';
import {
  createV4Policy,
  V4_DOCTRINES,
  V4_FIRE_MODES
} from './commander-v4-policy.js';

function parseArgs(argv) {
  const options = {
    scenarios: 24,
    workers: availableParallelism(),
    seed: 844733,
    output: '.training/commander-v4-teacher.json',
    opponents: ['crowd', 'commander_v3', 'ppo', 'neural', 'offensive', 'defensive', 'adaptive']
  };
  for (let index = 0; index < argv.length; index++) {
    if (argv[index] === '--scenarios') options.scenarios = Number(argv[++index]);
    else if (argv[index] === '--workers') options.workers = Number(argv[++index]);
    else if (argv[index] === '--seed') options.seed = Number(argv[++index]);
    else if (argv[index] === '--output') options.output = argv[++index];
    else if (argv[index] === '--opponents') {
      options.opponents = argv[++index].split(',').filter(Boolean);
    }
  }
  return options;
}

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function score(result, team, units) {
  const expected = team === 0 ? 'blue' : 'red';
  const win = result.winner === expected ? 1 : result.winner === 'draw' ? 0 : -1;
  const edge = (result.survivors[team] - result.survivors[1 - team]) / Math.max(1, units);
  return win + edge * 0.4 - Math.max(0, result.duration - 210) / 300;
}

function runPlan(scenario, doctrine, focus, fireMode, team, record = false) {
  const policy = createV4Policy();
  const blue = team === 0 ? 'commander_v4' : scenario.opponent;
  const red = team === 1 ? 'commander_v4' : scenario.opponent;
  const config = {
    forcedDoctrine: doctrine,
    forcedFocus: focus,
    forcedFire: fireMode,
    record
  };
  const training = team === 0 ? [config, null] : [null, config];
  return new Battle({
    blue, red, units: scenario.units, seed: scenario.seed,
    policies: team === 0 ? [policy, null] : [null, policy],
    training
  }).run();
}

function searchScenario(scenario) {
  const focusCandidates = [1, 3, 5];
  let best = null;
  for (let doctrine = 0; doctrine < V4_DOCTRINES.length; doctrine++) {
    for (const focus of focusCandidates) {
      for (let fireMode = 0; fireMode < 2; fireMode++) {
        let total = 0;
        for (let team = 0; team < 2; team++) {
          total += score(
            runPlan(scenario, doctrine, focus, fireMode, team),
            team,
            scenario.units
          );
        }
        if (!best || total > best.score) {
          best = { doctrine, focus, fireMode, score: total };
        }
      }
    }
  }
  const samples = [];
  for (let team = 0; team < 2; team++) {
    const result = runPlan(
      scenario, best.doctrine, best.focus, best.fireMode, team, true
    );
    const trajectory = result.trainingSamples[team];
    for (const sample of trajectory) {
      samples.push({
        features: sample.features,
        activeMask: sample.activeMask,
        hidden: sample.hidden,
        doctrine: best.doctrine,
        focus: best.focus,
        fireMode: best.fireMode,
        weight: Math.max(0.25, best.score + 2) / Math.max(1, trajectory.length)
      });
    }
  }
  return {
    samples,
    result: {
      ...scenario,
      doctrine: best.doctrine,
      doctrineName: V4_DOCTRINES[best.doctrine],
      focus: best.focus,
      fireMode: best.fireMode,
      fireModeName: V4_FIRE_MODES[best.fireMode],
      score: best.score
    }
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const random = rng(options.seed);
  const units = [80, 100, 140, 180, 240, 320];
  const scenarios = Array.from({ length: options.scenarios }, (_, index) => ({
    id: index,
    seed: ((options.seed + index * 104729 + random() * 1e8) | 0) >>> 0,
    units: units[(random() * units.length) | 0],
    opponent: options.opponents[index % options.opponents.length]
  }));
  const count = Math.max(1, Math.min(options.workers, scenarios.length));
  const chunks = Array.from({ length: count }, () => []);
  scenarios.forEach((scenario, index) => chunks[index % count].push(scenario));
  const parts = await Promise.all(chunks.map(chunk => new Promise((done, fail) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: { teacherWorker: true, scenarios: chunk }
    });
    worker.once('message', done);
    worker.once('error', fail);
  })));
  const results = parts.flatMap(part => part.results).sort((a, b) => a.id - b.id);
  const samples = parts.flatMap(part => part.samples);
  const output = resolve(options.output);
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify({ options, results, samples }), 'utf8');
  console.log(JSON.stringify({
    scenarios: results.length,
    searchedBattles: results.length * V4_DOCTRINES.length * 3 * 2 * 2,
    samples: samples.length,
    doctrines: Object.fromEntries(V4_DOCTRINES.map((name, index) => [
      name, results.filter(result => result.doctrine === index).length
    ]))
  }));
}

if (isMainThread) await main();
else if (workerData?.teacherWorker) {
  const results = workerData.scenarios.map(searchScenario);
  parentPort.postMessage({
    results: results.map(item => item.result),
    samples: results.flatMap(item => item.samples)
  });
}
