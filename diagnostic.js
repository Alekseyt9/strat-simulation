import {
  evaluateOrders,
  isCompatiblePolicy,
  POLICY_STANCES,
  SECTOR_COUNT
} from './neural-policy.js';
import { TRAINED_POLICY } from './trained-policy.js';
import {
  PPO_MEMORY,
  chooseCategorical,
  createPpoPolicy,
  evaluatePpoOrders,
  isCompatiblePpoPolicy,
  maskedDistribution
} from './ppo-policy.js';
import { TRAINED_PPO_POLICY } from './trained-ppo-policy.js';
import {
  V3_MEMORY,
  chooseV3Action,
  createV3Policy,
  evaluateV3Orders,
  isCompatibleV3Policy,
  mixedDistribution
} from './commander-v3-policy.js';
import { TRAINED_COMMANDER_V3_POLICY } from './trained-commander-v3-policy.js';
import {
  V4_DOCTRINES,
  V4_FIRE_MODES,
  V4_FOCUS_SECTORS,
  V4_MEMORY,
  chooseV4,
  createV4Policy,
  evaluateV4,
  isCompatibleV4Policy,
  v4Distribution
} from './commander-v4-policy.js';
import { TRAINED_COMMANDER_V4_POLICY } from './trained-commander-v4-policy.js';

// Shared deterministic CPU simulation core and headless diagnostic runner.
const isNodeRuntime = typeof process !== 'undefined' && Boolean(process.versions?.node);
let WorkerCtor;
let nodeIsMainThread = true;
let nodeParentPort;
let nodeWorkerData;
let cpuParallelism = 1;

const WORLD_W = 1600;
const WORLD_D = 900;
const REGIMENTS = 6;
const ARCHER_RATIO = 0.22;
const SHOT_CADENCE = 2.1;
const UNIT_RADIUS = 4.4;
const ATTACK_ARC_COS = Math.cos(Math.PI * 0.42);
const CELL = 30;
const GRID_W = Math.ceil(WORLD_W / CELL);
const GRID_D = Math.ceil(WORLD_D / CELL);
const DT = 1 / 30;
const MAX_TIME = 240;
const ROLES = ['line', 'line', 'line', 'archer', 'reserve', 'flank'];
const STRATEGIES = [
  'crowd', 'offensive', 'defensive', 'adaptive',
  'neural', 'ppo', 'commander_v3', 'commander_v4'
];
const STRATEGY_NAMES = {
  crowd: 'Без командования',
  offensive: 'Наступательная',
  defensive: 'Оборонительная',
  adaptive: 'Адаптивная',
  neural: 'Нейрокомандир (старый)',
  ppo: 'PPO-командир',
  commander_v3: 'Командир V3',
  commander_v4: 'Командир V4'
};

function parseArgs(argv) {
  const result = {
    blue: 'crowd',
    red: 'adaptive',
    units: 150,
    trials: 20,
    seed: 1,
    workers: cpuParallelism,
    blueFire: 'auto',
    redFire: 'auto',
    matrix: false,
    json: false
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--matrix') result.matrix = true;
    else if (arg === '--json') result.json = true;
    else if (arg === '--blue') result.blue = argv[++i];
    else if (arg === '--red') result.red = argv[++i];
    else if (arg === '--units') result.units = Number(argv[++i]);
    else if (arg === '--trials') result.trials = Number(argv[++i]);
    else if (arg === '--seed') result.seed = Number(argv[++i]);
    else if (arg === '--workers') result.workers = Number(argv[++i]);
    else if (arg === '--blue-fire') result.blueFire = argv[++i];
    else if (arg === '--red-fire') result.redFire = argv[++i];
  }
  if (!STRATEGIES.includes(result.blue) || !STRATEGIES.includes(result.red)) {
    throw new Error(`Стратегии: ${STRATEGIES.join(', ')}`);
  }
  if (!Number.isFinite(result.units) || result.units < 30 || result.units > 1200) {
    throw new Error('--units должен быть от 30 до 1200');
  }
  if (!Number.isFinite(result.trials) || result.trials < 1) {
    throw new Error('--trials должен быть положительным числом');
  }
  if (!Number.isFinite(result.workers) || result.workers < 1) {
    throw new Error('--workers должен быть положительным числом');
  }
  return result;
}

function makeRng(initialSeed) {
  let state = initialSeed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function angleDelta(from, to) {
  let delta = (to - from + Math.PI) % (Math.PI * 2) - Math.PI;
  if (delta < -Math.PI) delta += Math.PI * 2;
  return delta;
}

function turnTowards(current, target, maximum) {
  const delta = angleDelta(current, target);
  return current + Math.max(-maximum, Math.min(maximum, delta));
}

function shuffledRegiments(rand) {
  const values = Array.from({ length: REGIMENTS }, (_, index) => index);
  for (let i = values.length - 1; i > 0; i--) {
    const j = (rand() * (i + 1)) | 0;
    [values[i], values[j]] = [values[j], values[i]];
  }
  return values;
}

function createDisposition(team, rand) {
  const plans = [];
  for (let regiment = 0; regiment < REGIMENTS; regiment++) {
    const role = ROLES[regiment];
    const dir = team === 0 ? 1 : -1;
    let depth;
    let z;
    if (role === 'line') {
      depth = 380 + rand() * 125;
      z = [-220, 0, 220][regiment] + (rand() - 0.5) * 70;
    } else if (role === 'archer') {
      depth = 560 + rand() * 80;
      z = (rand() - 0.5) * 150;
    } else if (role === 'reserve') {
      depth = 620 + rand() * 55;
      z = (rand() - 0.5) * 180;
    } else {
      depth = 420 + rand() * 100;
      z = (rand() < 0.5 ? -1 : 1) * (285 + rand() * 45);
    }
    const x = -dir * depth;
    plans.push({
      x,
      z,
      homeX: x,
      homeZ: z,
      role,
      targetRegiment: 0,
      pace: 0.88 + rand() * 0.24,
      mission: 'advance',
      fireMode: 'independent',
      flankSign: rand() < 0.5 ? -1 : 1,
      formationCenterX: 0,
      initialCount: 0
    });
  }
  return plans;
}

function createCommander(team, mode, policy = null, training = null) {
  const defaultPolicy = mode === 'ppo'
    ? (isCompatiblePpoPolicy(TRAINED_PPO_POLICY) ? TRAINED_PPO_POLICY : createPpoPolicy())
    : mode === 'commander_v3'
      ? (isCompatibleV3Policy(TRAINED_COMMANDER_V3_POLICY)
        ? TRAINED_COMMANDER_V3_POLICY
        : createV3Policy())
      : mode === 'commander_v4'
        ? (isCompatibleV4Policy(TRAINED_COMMANDER_V4_POLICY)
          ? TRAINED_COMMANDER_V4_POLICY
          : createV4Policy())
      : TRAINED_POLICY;
  return {
    team,
    mode,
    nextDecision: 0,
    nextVolley: SHOT_CADENCE,
    volleyStart: -1,
    volleyUntil: -1,
    volleyId: 0,
    tactic: mode,
    policy: policy ?? defaultPolicy,
    policyOutputs: null,
    recurrentState: Array.from({ length: REGIMENTS }, () => new Array(PPO_MEMORY).fill(0)),
    v3Hidden: new Array(V3_MEMORY).fill(0),
    v4Hidden: new Array(V4_MEMORY).fill(0),
    training,
    decisions: {}
  };
}

function teamDefenseX(team) {
  return team === 0 ? -185 : 185;
}

function sectorCenter(index) {
  return -WORLD_D / 2 + (index + 0.5) * WORLD_D / SECTOR_COUNT;
}

function sectorIndex(z) {
  return Math.max(0, Math.min(
    SECTOR_COUNT - 1,
    Math.floor((z + WORLD_D / 2) / (WORLD_D / SECTOR_COUNT))
  ));
}

function finalizePpoRecords(records, terminalReward) {
  const gamma = 0.985;
  const gaeLambda = 0.95;
  const result = [];
  for (let regiment = 0; regiment < REGIMENTS; regiment++) {
    const trajectory = records.filter(record => record.regiment === regiment);
    let gae = 0;
    for (let index = trajectory.length - 1; index >= 0; index--) {
      const current = trajectory[index];
      const next = trajectory[index + 1];
      const done = !next;
      const potentialDelta = (next?.potential ?? current.potential) - current.potential;
      const reward = potentialDelta * 0.15 + (done ? terminalReward : 0);
      const nextValue = next?.value ?? 0;
      const delta = reward + gamma * nextValue * (done ? 0 : 1) - current.value;
      gae = delta + gamma * gaeLambda * (done ? 0 : 1) * gae;
      result.push({
        ...current,
        reward,
        done,
        advantage: gae,
        return: current.value + gae
      });
    }
  }
  return result.sort((a, b) => a.time - b.time || a.regiment - b.regiment);
}

function finalizeV3Records(records, terminalReward) {
  const gamma = 0.985;
  const gaeLambda = 0.95;
  const result = new Array(records.length);
  let gae = 0;
  for (let index = records.length - 1; index >= 0; index--) {
    const current = records[index];
    const next = records[index + 1];
    const done = !next;
    const potentialDelta = (next?.potential ?? current.potential) - current.potential;
    const reward = potentialDelta * 0.15 + (done ? terminalReward : 0);
    const nextValue = next?.value ?? 0;
    const delta = reward + gamma * nextValue * (done ? 0 : 1) - current.value;
    gae = delta + gamma * gaeLambda * (done ? 0 : 1) * gae;
    result[index] = {
      ...current,
      reward,
      done,
      advantage: gae,
      return: current.value + gae
    };
  }
  return result;
}

export class Battle {
  constructor({
    blue,
    red,
    units,
    seed,
    policies = [],
    training = [],
    fireModes = []
  }) {
    this.unitsPerArmy = units;
    this.rand = makeRng(seed);
    this.units = [];
    this.projectiles = [];
    this.time = 0;
    this.lastKillTime = 0;
    this.ended = false;
    this.plans = [createDisposition(0, this.rand), createDisposition(1, this.rand)];
    this.commanders = [
      createCommander(0, blue, policies[0], training[0]),
      createCommander(1, red, policies[1], training[1])
    ];
    this.commanders[0].fireModeOverride = fireModes[0] ?? 'auto';
    this.commanders[1].fireModeOverride = fireModes[1] ?? 'auto';
    this.trainingRecords = [[], []];
    this.metrics = [
      { arrows: 0, hits: 0, meleeHits: 0, kills: 0, decisions: 0 },
      { arrows: 0, hits: 0, meleeHits: 0, kills: 0, decisions: 0 }
    ];
    this.spawn();
  }

  spawn() {
    const count = this.unitsPerArmy;
    const archerTotal = Math.round(count * ARCHER_RATIO);
    const infantryTotal = count - archerTotal;
    const infantryWeights = [0.205, 0.205, 0.205, 0, 0.205, 0.18];
    const regimentCounts = new Array(REGIMENTS).fill(0);
    regimentCounts[3] = archerTotal;
    let assigned = 0;
    const infantryRegiments = [0, 1, 2, 4, 5];
    for (let index = 0; index < infantryRegiments.length; index++) {
      const regiment = infantryRegiments[index];
      const regimentCount = index === infantryRegiments.length - 1
        ? infantryTotal - assigned
        : Math.floor(infantryTotal * infantryWeights[regiment]);
      regimentCounts[regiment] = regimentCount;
      assigned += regimentCount;
    }

    const targets = [shuffledRegiments(this.rand), shuffledRegiments(this.rand)];
    for (let team = 0; team < 2; team++) {
      const dir = team === 0 ? 1 : -1;
      for (let regiment = 0; regiment < REGIMENTS; regiment++) {
        const plan = this.plans[team][regiment];
        plan.targetRegiment = targets[team][regiment];
        const regimentCount = regimentCounts[regiment];
        plan.initialCount = regimentCount;
        const rows = Math.max(6, Math.ceil(Math.sqrt(regimentCount * 1.15)));
        let offsetSumX = 0;
        for (let n = 0; n < regimentCount; n++) {
          const type = plan.role === 'archer' ? 1 : 0;
          const row = n % rows;
          const col = Math.floor(n / rows);
          const offsetX = -dir * col * 10;
          const offsetZ = (row - (rows - 1) / 2) * 10;
          const maxHp = type ? 38 : 60;
          this.units.push({
            team,
            type,
            regiment,
            x: plan.x + offsetX + (this.rand() - 0.5) * 2,
            z: plan.z + offsetZ + (this.rand() - 0.5) * 2,
            formationOffsetX: offsetX,
            formationOffsetZ: offsetZ,
            vx: dir * (12 + this.rand() * 4),
            vz: (this.rand() - 0.5) * 2,
            hp: maxHp * (0.78 + this.rand() * 0.22),
            cooldown: this.rand() * (type ? SHOT_CADENCE : 0.8),
            morale: 0.84 + this.rand() * 0.16,
            fatigue: this.rand() * 0.04,
            facing: team === 0 ? 0 : Math.PI,
            routing: false,
            alive: true,
            fall: 0,
            lastVolley: -1,
            volleyDelay: this.rand() * 0.55,
            phase: this.rand() * Math.PI * 2
          });
          offsetSumX += offsetX;
        }
        plan.formationCenterX = regimentCount ? offsetSumX / regimentCount : 0;
      }
    }
  }

  buildGrid() {
    const cells = new Array(GRID_W * GRID_D);
    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      if (!u.alive) continue;
      const gx = Math.max(0, Math.min(GRID_W - 1, ((u.x + WORLD_W / 2) / CELL) | 0));
      const gz = Math.max(0, Math.min(GRID_D - 1, ((u.z + WORLD_D / 2) / CELL) | 0));
      const index = gz * GRID_W + gx;
      if (!cells[index]) cells[index] = [];
      cells[index].push(i);
    }
    return cells;
  }

  regimentState() {
    const state = Array.from({ length: 2 }, () =>
      Array.from({ length: REGIMENTS }, () => ({ count: 0, x: 0, z: 0 }))
    );
    const teamIndices = [[], []];
    const regimentIndices = Array.from({ length: 2 }, () =>
      Array.from({ length: REGIMENTS }, () => [])
    );
    const alive = [0, 0];
    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      if (!u.alive) continue;
      alive[u.team]++;
      teamIndices[u.team].push(i);
      regimentIndices[u.team][u.regiment].push(i);
      const regiment = state[u.team][u.regiment];
      regiment.count++;
      regiment.x += u.x;
      regiment.z += u.z;
    }
    for (const army of state) {
      for (const regiment of army) {
        if (!regiment.count) continue;
        regiment.x /= regiment.count;
        regiment.z /= regiment.count;
      }
    }
    return { state, teamIndices, regimentIndices, alive };
  }

  createNeuralOrders(team, ownState, enemyState, alive, enemyAlive, closestDistance, commander) {
    const sectors = Array.from({ length: 2 }, () =>
      Array.from({ length: SECTOR_COUNT }, () => ({
        count: 0,
        archers: 0,
        morale: 0,
        fatigue: 0,
        x: 0
      }))
    );
    const regimentCondition = Array.from({ length: REGIMENTS }, () => ({
      count: 0,
      morale: 0,
      fatigue: 0
    }));
    let ownMorale = 0;
    let ownFatigue = 0;
    for (const unit of this.units) {
      if (!unit.alive) continue;
      const sector = sectors[unit.team][sectorIndex(unit.z)];
      sector.count++;
      sector.archers += unit.type;
      sector.morale += unit.morale;
      sector.fatigue += unit.fatigue;
      sector.x += unit.x;
      if (unit.team === team) {
        const condition = regimentCondition[unit.regiment];
        condition.count++;
        condition.morale += unit.morale;
        condition.fatigue += unit.fatigue;
        ownMorale += unit.morale;
        ownFatigue += unit.fatigue;
      }
    }
    for (const army of sectors) {
      for (const sector of army) {
        if (!sector.count) continue;
        sector.morale /= sector.count;
        sector.fatigue /= sector.count;
        sector.x /= sector.count;
      }
    }

    const livingOwn = ownState.filter(state => state.count);
    const livingEnemy = enemyState.filter(state => state.count);
    const ownWidth = livingOwn.length
      ? Math.max(...livingOwn.map(state => state.z)) - Math.min(...livingOwn.map(state => state.z))
      : 0;
    const enemyWidth = livingEnemy.length
      ? Math.max(...livingEnemy.map(state => state.z)) - Math.min(...livingEnemy.map(state => state.z))
      : 0;
    const relativeStrength = alive / Math.max(1, enemyAlive);
    const assigned = new Array(SECTOR_COUNT).fill(0);
    const usedLineSectors = new Set();
    const orders = new Array(REGIMENTS);
    const roleIndex = new Map(ROLES.map((role, index) => [role, index]));
    const allowedByRole = {
      line: ['hold', 'advance', 'assault'],
      archer: ['hold', 'advance'],
      reserve: ['reserve', 'hold', 'assault'],
      flank: ['hold', 'advance', 'assault', 'flank']
    };
    const processingOrder = [0, 1, 2, 5, 4, 3];
    const forcePursuit = (
      commander.mode === 'ppo'
      || commander.mode === 'commander_v3'
      || commander.mode === 'commander_v4'
    )
      && (this.time >= 120 || (
        this.time >= 45
        && this.time - this.lastKillTime >= 18
      ));
    commander.pursuit = forcePursuit;
    const makeFeatures = regiment => {
      const own = ownState[regiment];
      const plan = this.plans[team][regiment];
      const condition = regimentCondition[regiment];
      const currentSector = sectorIndex(own.z);
      const features = [
        Math.max(-1, Math.min(1, relativeStrength - 1)),
        Math.min(1, closestDistance / 800),
        Math.min(1, this.time / MAX_TIME),
        Math.min(1, ownWidth / WORLD_D),
        Math.min(1, enemyWidth / WORLD_D),
        alive / this.unitsPerArmy,
        enemyAlive / this.unitsPerArmy,
        alive ? ownMorale / alive * 2 - 1 : -1,
        alive ? ownFatigue / alive * 2 - 1 : 1,
        own.count / Math.max(1, plan.initialCount),
        ...ROLES.map(role => role === plan.role ? 1 : 0),
        condition.count ? condition.morale / condition.count * 2 - 1 : -1,
        condition.count ? condition.fatigue / condition.count * 2 - 1 : 1,
        currentSector / (SECTOR_COUNT - 1) * 2 - 1,
        (own.x * (team === 0 ? 1 : -1)) / (WORLD_W / 2)
      ];
      for (let index = 0; index < SECTOR_COUNT; index++) {
        const ownSector = sectors[team][index];
        const enemySector = sectors[1 - team][index];
        features.push(
          ownSector.count / Math.max(1, alive),
          enemySector.count / Math.max(1, enemyAlive),
          Math.max(-1, Math.min(1, (ownSector.count - enemySector.count) / Math.max(1, this.unitsPerArmy * 0.25))),
          enemySector.count ? enemySector.archers / enemySector.count : 0
        );
      }
      return features;
    };

    if (commander.mode === 'commander_v4') {
      const activeMask = ownState.map(state => state.count > 0);
      const activeRegimentRatio = activeMask.filter(Boolean).length / Math.max(1, activeMask.length);
      const regimentFeatures = ownState.map((state, regiment) => {
        if (!state.count) return new Array(48).fill(0);
        const legacy = makeFeatures(regiment);
        const role = this.plans[team][regiment].role;
        return [
          ...legacy.slice(0, 10),
          ...['line', 'archer', 'reserve', 'flank'].map(candidate => candidate === role ? 1 : 0),
          regiment / Math.max(1, ownState.length - 1) * 2 - 1,
          activeRegimentRatio,
          ...legacy.slice(16)
        ];
      });
      const previousHidden = commander.v4Hidden;
      const network = evaluateV4(regimentFeatures, activeMask, previousHidden, commander.policy);
      commander.v4Hidden = network.hidden;
      const exploration = commander.training?.sample
        ? Math.max(0, Math.min(0.4, commander.training.exploration ?? 0.1))
        : 0;
      const doctrineProbabilities = v4Distribution(network.doctrines, exploration);
      const focusProbabilities = v4Distribution(network.focus, exploration);
      const fireProbabilities = v4Distribution(network.fire, exploration);
      const samplingRng = commander.training?.sample ? this.rand : null;
      let doctrineIndex = chooseV4(doctrineProbabilities, samplingRng);
      let focusSector = chooseV4(focusProbabilities, samplingRng);
      let fireModeIndex = chooseV4(fireProbabilities, samplingRng);
      if (Number.isInteger(commander.training?.forcedDoctrine)) {
        doctrineIndex = Math.max(
          0,
          Math.min(V4_DOCTRINES.length - 1, commander.training.forcedDoctrine)
        );
      }
      if (Number.isInteger(commander.training?.forcedFocus)) {
        focusSector = Math.max(
          0,
          Math.min(V4_FOCUS_SECTORS - 1, commander.training.forcedFocus)
        );
      }
      if (Number.isInteger(commander.training?.forcedFire)) {
        fireModeIndex = Math.max(
          0,
          Math.min(V4_FIRE_MODES.length - 1, commander.training.forcedFire)
        );
      }
      if (
        commander.fireModeOverride
        && commander.fireModeOverride !== 'auto'
        && V4_FIRE_MODES.includes(commander.fireModeOverride)
      ) {
        fireModeIndex = V4_FIRE_MODES.indexOf(commander.fireModeOverride);
      }
      if (forcePursuit) {
        doctrineIndex = V4_DOCTRINES.indexOf('mass_assault');
        fireModeIndex = V4_FIRE_MODES.indexOf('independent');
      }
      const doctrine = V4_DOCTRINES[doctrineIndex];
      const fireMode = V4_FIRE_MODES[fireModeIndex];
      if (doctrine === 'left_hook') focusSector = 1;
      else if (doctrine === 'right_hook') focusSector = 5;
      const lineRegiments = this.plans[team]
        .map((plan, regiment) => ({ plan, regiment }))
        .filter(item => item.plan.role === 'line')
        .map(item => item.regiment);

      for (let regiment = 0; regiment < REGIMENTS; regiment++) {
        const own = ownState[regiment];
        if (!own.count) continue;
        const plan = this.plans[team][regiment];
        if (plan.role === 'archer') plan.fireMode = fireMode;
        let targetSector = focusSector;
        let stance = 'hold';
        if (doctrine === 'mass_advance') {
          stance = plan.role === 'archer' ? 'hold' : 'advance';
        } else if (doctrine === 'mass_assault') {
          stance = plan.role === 'archer' ? 'advance' : 'assault';
        } else if (doctrine === 'elastic') {
          if (plan.role === 'line') {
            const lineIndex = lineRegiments.indexOf(regiment);
            targetSector = Math.max(0, Math.min(6, focusSector + lineIndex - 1));
          } else if (plan.role === 'reserve') {
            stance = 'reserve';
          } else if (plan.role === 'flank') {
            targetSector = plan.homeZ < 0 ? 1 : 5;
          }
        } else if (doctrine === 'left_hook' || doctrine === 'right_hook') {
          stance = plan.role === 'archer'
            ? 'hold'
            : plan.role === 'flank' ? 'flank' : plan.role === 'reserve' ? 'assault' : 'advance';
        } else if (doctrine === 'counterattack') {
          stance = closestDistance < 210
            ? (plan.role === 'archer' ? 'advance' : 'assault')
            : plan.role === 'reserve' ? 'reserve' : 'hold';
        } else if (doctrine === 'encircle') {
          if (plan.role === 'flank') {
            stance = 'flank';
            targetSector = plan.homeZ < 0 ? 0 : 6;
          } else if (plan.role === 'reserve') {
            stance = closestDistance < 250 ? 'assault' : 'reserve';
          } else {
            stance = 'hold';
          }
        }

        let targetRegiment = 0;
        let targetDistance = Infinity;
        for (let enemyRegiment = 0; enemyRegiment < REGIMENTS; enemyRegiment++) {
          const enemy = enemyState[enemyRegiment];
          if (!enemy.count) continue;
          const distance = forcePursuit
            ? Math.hypot(enemy.x - own.x, enemy.z - own.z)
            : Math.abs(enemy.z - sectorCenter(targetSector));
          if (distance < targetDistance) {
            targetDistance = distance;
            targetRegiment = enemyRegiment;
          }
        }
        if (forcePursuit && enemyState[targetRegiment]?.count) {
          targetSector = sectorIndex(enemyState[targetRegiment].z);
        }
        orders[regiment] = { stance, targetSector, targetRegiment, score: 0 };
      }
      commander.decisions[doctrine] = (commander.decisions[doctrine] ?? 0) + 1;
      commander.currentDoctrine = doctrine;
      commander.currentFireMode = fireMode;
      if (commander.training?.record) {
        let isolatedEngagements = 0;
        let engagedRegiments = 0;
        for (let regiment = 0; regiment < REGIMENTS; regiment++) {
          const own = ownState[regiment];
          if (!own.count || this.plans[team][regiment].role === 'archer') continue;
          const enemyDistance = enemyState
            .filter(state => state.count)
            .reduce(
              (best, state) => Math.min(best, Math.hypot(state.x - own.x, state.z - own.z)),
              Infinity
            );
          if (enemyDistance >= 145) continue;
          engagedRegiments++;
          const supported = ownState.some((ally, allyRegiment) =>
            allyRegiment !== regiment
            && ally.count
            && this.plans[team][allyRegiment].role !== 'archer'
            && Math.hypot(ally.x - own.x, ally.z - own.z) < 190
          );
          if (!supported) isolatedEngagements++;
        }
        const isolationRatio = isolatedEngagements / Math.max(1, engagedRegiments);
        this.trainingRecords[team].push({
          kind: 'commander_v4',
          time: this.time,
          features: regimentFeatures,
          activeMask,
          hidden: previousHidden.slice(),
          doctrine: doctrineIndex,
          focus: focusSector,
          fireMode: fireModeIndex,
          doctrineLogProb: Math.log(Math.max(1e-9, doctrineProbabilities[doctrineIndex])),
          focusLogProb: Math.log(Math.max(1e-9, focusProbabilities[focusSector])),
          fireLogProb: Math.log(Math.max(1e-9, fireProbabilities[fireModeIndex])),
          logProb: Math.log(Math.max(1e-9, doctrineProbabilities[doctrineIndex]))
            + Math.log(Math.max(1e-9, focusProbabilities[focusSector]))
            + Math.log(Math.max(1e-9, fireProbabilities[fireModeIndex])),
          value: network.value,
          potential: (alive - enemyAlive) / Math.max(1, this.unitsPerArmy)
            - isolationRatio * 0.12
        });
      }
      return orders;
    }

    if (commander.mode === 'commander_v3') {
      const activeMask = ownState.map(state => state.count > 0);
      const activeRegimentRatio = activeMask.filter(Boolean).length / Math.max(1, activeMask.length);
      const regimentFeatures = ownState.map((state, regiment) => {
        if (!state.count) return new Array(48).fill(0);
        const legacy = makeFeatures(regiment);
        const role = this.plans[team][regiment].role;
        return [
          ...legacy.slice(0, 10),
          ...['line', 'archer', 'reserve', 'flank'].map(candidate => candidate === role ? 1 : 0),
          regiment / Math.max(1, ownState.length - 1) * 2 - 1,
          activeRegimentRatio,
          ...legacy.slice(16)
        ];
      });
      const previousHidden = commander.v3Hidden;
      const network = evaluateV3Orders(
        regimentFeatures,
        activeMask,
        previousHidden,
        commander.policy
      );
      commander.v3Hidden = network.hidden;
      const actions = {
        sectors: new Array(REGIMENTS).fill(-1),
        stances: new Array(REGIMENTS).fill(-1),
        sectorMasks: Array.from({ length: REGIMENTS }, () => []),
        stanceMasks: Array.from({ length: REGIMENTS }, () => []),
        actionLogProbs: new Array(REGIMENTS).fill(0)
      };
      let jointLogProb = 0;
      const exploration = commander.training?.sample
        ? Math.max(0, Math.min(0.4, commander.training.exploration ?? 0.08))
        : 0;
      for (const regiment of processingOrder) {
        const own = ownState[regiment];
        if (!own.count) continue;
        const plan = this.plans[team][regiment];
        // V3 may deliberately concentrate several line regiments in one sector.
        // Older policies keep their uniqueness constraint for direct comparison.
        let availableSectors = Array.from({ length: SECTOR_COUNT }, (_, index) => index);
        let allowedStances = allowedByRole[plan.role].map(stance => POLICY_STANCES.indexOf(stance));
        if (forcePursuit) {
          const nearestEnemy = enemyState
            .filter(state => state.count)
            .reduce((nearest, state) => {
              const distance = Math.hypot(state.x - own.x, state.z - own.z);
              return !nearest || distance < nearest.distance ? { ...state, distance } : nearest;
            }, null);
          if (nearestEnemy) availableSectors = [sectorIndex(nearestEnemy.z)];
          allowedStances = [
            POLICY_STANCES.indexOf(plan.role === 'archer' ? 'advance' : 'assault')
          ];
        }
        const sectorProbabilities = mixedDistribution(
          network.outputs[regiment].sectors,
          availableSectors,
          exploration
        );
        const stanceProbabilities = mixedDistribution(
          network.outputs[regiment].stances,
          allowedStances,
          exploration
        );
        const samplingRng = commander.training?.sample ? this.rand : null;
        const targetSector = chooseV3Action(sectorProbabilities, samplingRng);
        const stanceIndex = chooseV3Action(stanceProbabilities, samplingRng);
        const stance = POLICY_STANCES[stanceIndex];
        const targetZ = sectorCenter(targetSector);
        let targetRegiment = 0;
        let targetDistance = Infinity;
        for (let enemyRegiment = 0; enemyRegiment < REGIMENTS; enemyRegiment++) {
          const enemy = enemyState[enemyRegiment];
          if (!enemy.count) continue;
          const distance = Math.abs(enemy.z - targetZ);
          if (distance < targetDistance) {
            targetDistance = distance;
            targetRegiment = enemyRegiment;
          }
        }
        orders[regiment] = { stance, targetSector, targetRegiment, score: 0 };
        actions.sectors[regiment] = targetSector;
        actions.stances[regiment] = stanceIndex;
        actions.sectorMasks[regiment] = availableSectors;
        actions.stanceMasks[regiment] = allowedStances;
        const actionLogProb = Math.log(Math.max(1e-9, sectorProbabilities[targetSector]))
          + Math.log(Math.max(1e-9, stanceProbabilities[stanceIndex]));
        actions.actionLogProbs[regiment] = actionLogProb;
        jointLogProb += actionLogProb;
        commander.decisions[stance] = (commander.decisions[stance] ?? 0) + 1;
      }
      if (commander.training?.record) {
        let isolatedEngagements = 0;
        let engagedRegiments = 0;
        for (let regiment = 0; regiment < REGIMENTS; regiment++) {
          const own = ownState[regiment];
          if (!own.count || this.plans[team][regiment].role === 'archer') continue;
          const enemyDistance = enemyState
            .filter(state => state.count)
            .reduce(
              (best, state) => Math.min(best, Math.hypot(state.x - own.x, state.z - own.z)),
              Infinity
            );
          if (enemyDistance >= 145) continue;
          engagedRegiments++;
          const supported = ownState.some((ally, allyRegiment) =>
            allyRegiment !== regiment
            && ally.count
            && this.plans[team][allyRegiment].role !== 'archer'
            && Math.hypot(ally.x - own.x, ally.z - own.z) < 190
          );
          if (!supported) isolatedEngagements++;
        }
        const isolationRatio = isolatedEngagements / Math.max(1, engagedRegiments);
        this.trainingRecords[team].push({
          kind: 'commander_v3',
          time: this.time,
          features: regimentFeatures,
          activeMask,
          hidden: previousHidden.slice(),
          ...actions,
          logProb: jointLogProb,
          value: network.value,
          potential: (alive - enemyAlive) / Math.max(1, this.unitsPerArmy)
            - isolationRatio * 0.12
        });
      }
      return orders;
    }

    for (const regiment of processingOrder) {
      const own = ownState[regiment];
      if (!own.count) continue;
      const plan = this.plans[team][regiment];
      const condition = regimentCondition[regiment];
      const currentSector = sectorIndex(own.z);
      const preferredSector = plan.role === 'line'
        ? [1, 3, 5][regiment]
        : plan.role === 'flank'
          ? (plan.homeZ < 0 ? 0 : 6)
          : 3;
      let best = null;
      const features = makeFeatures(regiment);
      if (commander.mode === 'ppo') {
        const previousHidden = commander.recurrentState[regiment];
        const output = evaluatePpoOrders(features, previousHidden, commander.policy);
        commander.recurrentState[regiment] = output.hidden;
        let availableSectors = Array.from({ length: SECTOR_COUNT }, (_, index) => index)
          .filter(index => plan.role !== 'line' || !usedLineSectors.has(index));
        let allowedStances = allowedByRole[plan.role].map(stance => POLICY_STANCES.indexOf(stance));
        if (forcePursuit) {
          const nearestEnemy = enemyState
            .map((state, enemyRegiment) => ({ ...state, enemyRegiment }))
            .filter(state => state.count)
            .reduce((nearest, state) => {
              const distance = Math.hypot(state.x - own.x, state.z - own.z);
              return !nearest || distance < nearest.distance ? { ...state, distance } : nearest;
            }, null);
          if (nearestEnemy) {
            const pursuitSector = sectorIndex(nearestEnemy.z);
            availableSectors = [pursuitSector];
          }
          const pursuitStance = plan.role === 'archer' ? 'advance' : 'assault';
          allowedStances = [POLICY_STANCES.indexOf(pursuitStance)];
        }
        const sectorProbabilities = maskedDistribution(output.sectors, availableSectors);
        const stanceProbabilities = maskedDistribution(output.stances, allowedStances);
        const samplingRng = commander.training?.sample ? this.rand : null;
        const targetSector = chooseCategorical(sectorProbabilities, samplingRng);
        const stanceIndex = chooseCategorical(stanceProbabilities, samplingRng);
        const stance = POLICY_STANCES[stanceIndex];
        if (plan.role === 'line') usedLineSectors.add(targetSector);
        assigned[targetSector] += own.count;

        const targetZ = sectorCenter(targetSector);
        let targetRegiment = 0;
        let targetDistance = Infinity;
        for (let enemyRegiment = 0; enemyRegiment < REGIMENTS; enemyRegiment++) {
          const enemy = enemyState[enemyRegiment];
          if (!enemy.count) continue;
          const distance = Math.abs(enemy.z - targetZ);
          if (distance < targetDistance) {
            targetDistance = distance;
            targetRegiment = enemyRegiment;
          }
        }
        orders[regiment] = { stance, targetSector, targetRegiment, score: 0 };
        if (commander.training?.record) {
          this.trainingRecords[team].push({
            kind: 'ppo',
            regiment,
            time: this.time,
            features,
            hidden: previousHidden.slice(),
            sector: targetSector,
            stance: stanceIndex,
            sectorMask: availableSectors,
            stanceMask: allowedStances,
            logProb: Math.log(Math.max(1e-9, sectorProbabilities[targetSector]))
              + Math.log(Math.max(1e-9, stanceProbabilities[stanceIndex])),
            value: output.value,
            potential: (alive - enemyAlive) / Math.max(1, this.unitsPerArmy)
          });
        }
        commander.decisions[stance] = (commander.decisions[stance] ?? 0) + 1;
        continue;
      }
      const networkScores = isCompatiblePolicy(commander.policy)
        ? evaluateOrders(features, commander.policy)
        : null;
      for (let targetSector = 0; targetSector < SECTOR_COUNT; targetSector++) {
        const enemySector = sectors[1 - team][targetSector];
        for (const stance of allowedByRole[plan.role]) {
          let score = networkScores
            ? networkScores.sectors[targetSector] + networkScores.stances[POLICY_STANCES.indexOf(stance)]
            : 0;
          score -= Math.abs(targetSector - preferredSector) * 0.11;
          score -= assigned[targetSector] / Math.max(1, alive) * 0.9;
          if (!enemySector.count && (stance === 'assault' || stance === 'flank')) score -= 0.45;
          if (plan.role === 'archer' && stance === 'hold') score += 0.12;
          if (plan.role === 'reserve' && stance === 'reserve') score += 0.16;
          if (plan.role === 'flank' && stance === 'flank') score += 0.08;
          if (plan.role === 'line' && usedLineSectors.has(targetSector)) score -= 100;
          if (commander.training?.temperature > 0) {
            const random = Math.max(1e-7, Math.min(1 - 1e-7, this.rand()));
            score += -Math.log(-Math.log(random)) * commander.training.temperature;
          }
          if (!best || score > best.score) best = { stance, targetSector, score };
        }
      }
      if (commander.training?.epsilon > 0 && this.rand() < commander.training.epsilon) {
        const availableSectors = Array.from({ length: SECTOR_COUNT }, (_, index) => index)
          .filter(index => plan.role !== 'line' || !usedLineSectors.has(index));
        best = {
          targetSector: availableSectors[(this.rand() * availableSectors.length) | 0],
          stance: allowedByRole[plan.role][(this.rand() * allowedByRole[plan.role].length) | 0],
          score: 0
        };
      }
      if (plan.role === 'line') usedLineSectors.add(best.targetSector);
      assigned[best.targetSector] += own.count;
      const targetZ = sectorCenter(best.targetSector);
      let targetRegiment = 0;
      let targetDistance = Infinity;
      for (let enemyRegiment = 0; enemyRegiment < REGIMENTS; enemyRegiment++) {
        const enemy = enemyState[enemyRegiment];
        if (!enemy.count) continue;
        const distance = Math.abs(enemy.z - targetZ);
        if (distance < targetDistance) {
          targetDistance = distance;
          targetRegiment = enemyRegiment;
        }
      }
      orders[regiment] = { ...best, targetRegiment };
      if (commander.training?.record) {
        this.trainingRecords[team].push({
          features,
          sector: best.targetSector,
          stance: POLICY_STANCES.indexOf(best.stance)
        });
      }
      commander.decisions[best.stance] = (commander.decisions[best.stance] ?? 0) + 1;
    }
    return orders;
  }

  issueOrders(team, ownState, enemyState, alive) {
    const commander = this.commanders[team];
    if (this.time < commander.nextDecision) return;
    commander.nextDecision = this.time + (
      commander.mode === 'neural'
      || commander.mode === 'ppo'
      || commander.mode === 'commander_v3'
      || commander.mode === 'commander_v4'
        ? 4
        : 8
    );
    this.metrics[team].decisions++;

    const enemies = enemyState
      .map((state, regiment) => ({ ...state, regiment }))
      .filter(state => state.count > 0);
    if (!enemies.length) return;

    if (commander.mode === 'crowd') {
      commander.tactic = 'crowd';
      commander.label = 'БЕЗ КОМАНДОВАНИЯ';
      for (let regiment = 0; regiment < REGIMENTS; regiment++) {
        const own = ownState[regiment];
        if (!own.count) continue;
        let nearest = enemies[0];
        let best = Infinity;
        for (const enemy of enemies) {
          const distance = Math.hypot(enemy.x - own.x, enemy.z - own.z);
          if (distance < best) {
            best = distance;
            nearest = enemy;
          }
        }
        const plan = this.plans[team][regiment];
        plan.mission = 'crowd';
        plan.targetRegiment = nearest.regiment;
      }
      return;
    }

    const enemyAlive = enemies.reduce((sum, state) => sum + state.count, 0);
    const relativeStrength = alive / Math.max(1, enemyAlive);
    let closestThreat = enemies[0];
    let closestDistance = Infinity;
    for (const own of ownState) {
      if (!own.count) continue;
      for (const enemy of enemies) {
        const distance = Math.hypot(enemy.x - own.x, enemy.z - own.z);
        if (distance < closestDistance) {
          closestDistance = distance;
          closestThreat = enemy;
        }
      }
    }

    let doctrine = commander.mode;
    let neuralOrders = null;
    if (doctrine === 'adaptive') {
      doctrine = relativeStrength < 0.92 || (closestDistance < 190 && relativeStrength < 1.08)
        ? 'defensive'
        : 'offensive';
    } else if (
      doctrine === 'neural'
      || doctrine === 'ppo'
      || doctrine === 'commander_v3'
      || doctrine === 'commander_v4'
    ) {
      neuralOrders = this.createNeuralOrders(
        team,
        ownState,
        enemyState,
        alive,
        enemyAlive,
        closestDistance,
        commander
      );
      doctrine = commander.mode === 'commander_v4'
        ? 'commander_v4_detail'
        : commander.mode === 'commander_v3'
        ? 'commander_v3_detail'
        : commander.mode === 'ppo' ? 'ppo_detail' : 'neural_detail';
    }
    commander.tactic = doctrine;
    commander.label = commander.mode === 'adaptive'
      ? `АДАПТИВНО: ${doctrine === 'offensive' ? 'НАСТУПЛЕНИЕ' : 'ОБОРОНА'}`
      : commander.mode === 'neural'
        ? `НЕЙРОСЕТЬ (СТАРАЯ): ПРИКАЗЫ ПО СЕКТОРАМ`
        : commander.mode === 'ppo'
          ? commander.pursuit
            ? `PPO + GRU: ПРЕСЛЕДОВАНИЕ`
            : `PPO + GRU: ПРИКАЗЫ ПО СЕКТОРАМ`
          : commander.mode === 'commander_v3'
            ? commander.pursuit
              ? `V3 ATTENTION: ПРЕСЛЕДОВАНИЕ`
              : `V3 ATTENTION: СОВМЕСТНЫЙ ПЛАН`
            : commander.mode === 'commander_v4'
              ? commander.pursuit
                ? `V4 LEAGUE: ПРЕСЛЕДОВАНИЕ`
                : `V4 LEAGUE: ${String(commander.currentDoctrine ?? 'PLAN').toUpperCase()}`
        : doctrine === 'offensive' ? 'НАСТУПЛЕНИЕ ПО РОЛЯМ' : 'ЭШЕЛОНИРОВАННАЯ ОБОРОНА';
    if (
      commander.mode !== 'neural'
      && commander.mode !== 'ppo'
      && commander.mode !== 'commander_v3'
      && commander.mode !== 'commander_v4'
    ) {
      commander.decisions[doctrine] = (commander.decisions[doctrine] ?? 0) + 1;
    }

    const assaultRegiments = [ownState[1], ownState[2], ownState[4]].filter(state => state.count);
    const assaultCenter = assaultRegiments.reduce(
      (center, state) => ({ x: center.x + state.x / assaultRegiments.length, z: center.z + state.z / assaultRegiments.length }),
      { x: 0, z: 0 }
    );
    const assaultTarget = enemies.reduce((best, current) => {
      const bestScore = Math.hypot(best.x - assaultCenter.x, best.z - assaultCenter.z) + best.count * 2;
      const currentScore = Math.hypot(current.x - assaultCenter.x, current.z - assaultCenter.z) + current.count * 2;
      return currentScore < bestScore ? current : best;
    });
    const lowFlank = enemies.reduce((a, b) => b.z < a.z ? b : a);
    const highFlank = enemies.reduce((a, b) => b.z > a.z ? b : a);
    const breached = enemies.filter(enemy => team === 0 ? enemy.x < -120 : enemy.x > 120);

    for (let regiment = 0; regiment < REGIMENTS; regiment++) {
      const own = ownState[regiment];
      if (!own.count) continue;
      const plan = this.plans[team][regiment];
      let nearest = enemies[0];
      let best = Infinity;
      for (const enemy of enemies) {
        const distance = Math.hypot(enemy.x - own.x, enemy.z - own.z);
        if (distance < best) {
          best = distance;
          nearest = enemy;
        }
      }

      if (neuralOrders?.[regiment]) {
        const order = neuralOrders[regiment];
        plan.mission = `neural_${order.stance}`;
        plan.targetSector = order.targetSector;
        plan.targetRegiment = order.targetRegiment;
      } else if (doctrine === 'offensive') {
        if (plan.role === 'line') {
          plan.mission = regiment < 2 ? 'pin' : 'assault';
          plan.targetRegiment = regiment < 2 ? nearest.regiment : assaultTarget.regiment;
        } else if (plan.role === 'archer') {
          plan.mission = 'fire_support';
          plan.targetRegiment = assaultTarget.regiment;
        } else if (plan.role === 'reserve') {
          plan.mission = 'assault';
          plan.targetRegiment = assaultTarget.regiment;
        } else {
          plan.mission = 'flank';
          plan.flankSign = plan.homeZ < 0 ? -1 : 1;
          plan.targetRegiment = (plan.flankSign < 0 ? lowFlank : highFlank).regiment;
        }
      } else if (doctrine === 'elastic') {
        if (plan.role === 'line') {
          plan.mission = 'wide_screen';
          plan.targetRegiment = nearest.regiment;
        } else if (plan.role === 'archer') {
          plan.mission = 'defensive_fire';
          plan.targetRegiment = closestThreat.regiment;
        } else if (plan.role === 'reserve') {
          plan.mission = closestDistance < 330 ? 'counterattack' : 'reserve';
          plan.targetRegiment = closestThreat.regiment;
        } else {
          plan.mission = closestDistance < 280 ? 'counterattack' : 'wide_guard';
          plan.targetRegiment = (plan.homeZ < 0 ? lowFlank : highFlank).regiment;
        }
      } else if (plan.role === 'line') {
        plan.mission = 'screen';
        plan.targetRegiment = nearest.regiment;
      } else if (plan.role === 'archer') {
        plan.mission = 'defensive_fire';
        plan.targetRegiment = closestThreat.regiment;
      } else if (plan.role === 'reserve') {
        plan.mission = breached.length || closestDistance < 360 ? 'counterattack' : 'reserve';
        plan.targetRegiment = (breached[0] ?? closestThreat).regiment;
      } else {
        plan.mission = closestDistance < 380 ? 'counterattack' : 'guard';
        plan.targetRegiment = (plan.homeZ < 0 ? lowFlank : highFlank).regiment;
      }
    }
  }

  updateCommanders(state, alive) {
    this.issueOrders(0, state[0], state[1], alive[0]);
    this.issueOrders(1, state[1], state[0], alive[1]);
    for (const commander of this.commanders) {
      if (commander.mode === 'crowd') continue;
      if (
        commander.fireModeOverride
        && commander.fireModeOverride !== 'auto'
        && V4_FIRE_MODES.includes(commander.fireModeOverride)
      ) {
        for (const plan of this.plans[commander.team]) {
          if (plan.role === 'archer') plan.fireMode = commander.fireModeOverride;
        }
      }
      const usesVolley = this.plans[commander.team].some(
        plan => plan.role === 'archer' && plan.fireMode === 'volley'
      );
      if (!usesVolley) {
        commander.nextVolley = Math.max(commander.nextVolley, this.time + SHOT_CADENCE);
        commander.volleyUntil = -1;
        continue;
      }
      if (this.time >= commander.nextVolley) {
        commander.volleyId++;
        commander.volleyStart = this.time;
        commander.volleyUntil = this.time + 0.75;
        commander.nextVolley = this.time + SHOT_CADENCE;
      }
    }
  }

  kill(unit, killerTeam) {
    if (!unit.alive) return;
    unit.alive = false;
    unit.fall = 0.01;
    this.lastKillTime = this.time;
    if (unit.team !== killerTeam) this.metrics[killerTeam].kills++;
  }

  fireArrow(shooter, targetIndex, target) {
    const dx = target.x - shooter.x;
    const dz = target.z - shooter.z;
    const distance = Math.hypot(dx, dz);
    const flight = distance / 430;
    this.projectiles.push({
      x: shooter.x,
      y: 11,
      z: shooter.z,
      vx: dx / flight,
      vy: 68 + distance * 0.025,
      vz: dz / flight,
      life: flight,
      target: targetIndex,
      team: shooter.team,
      damage: 8 + this.rand() * 8
    });
    this.metrics[shooter.team].arrows++;
  }

  updateProjectiles(grid) {
    for (let i = this.projectiles.length - 1; i >= 0; i--) {
      const projectile = this.projectiles[i];
      projectile.life -= DT;
      projectile.x += projectile.vx * DT;
      projectile.y += projectile.vy * DT;
      projectile.z += projectile.vz * DT;
      projectile.vy -= 170 * DT;
      if (projectile.life > 0 && projectile.y >= 0) continue;
      let impactIndex = -1;
      let impactD2 = 90;
      const intended = this.units[projectile.target];
      if (intended?.alive) {
        const dx = intended.x - projectile.x;
        const dz = intended.z - projectile.z;
        const d2 = dx * dx + dz * dz;
        if (d2 < impactD2) {
          impactD2 = d2;
          impactIndex = projectile.target;
        }
      }
      const gx = Math.max(0, Math.min(GRID_W - 1, ((projectile.x + WORLD_W / 2) / CELL) | 0));
      const gz = Math.max(0, Math.min(GRID_D - 1, ((projectile.z + WORLD_D / 2) / CELL) | 0));
      for (let oz = -1; oz <= 1; oz++) {
        const cz = gz + oz;
        if (cz < 0 || cz >= GRID_D) continue;
        for (let ox = -1; ox <= 1; ox++) {
          const cx = gx + ox;
          if (cx < 0 || cx >= GRID_W) continue;
          const bucket = grid[cz * GRID_W + cx];
          if (!bucket) continue;
          for (const candidateIndex of bucket) {
            const candidate = this.units[candidateIndex];
            if (!candidate.alive) continue;
            const dx = candidate.x - projectile.x;
            const dz = candidate.z - projectile.z;
            const d2 = dx * dx + dz * dz;
            if (d2 < impactD2) {
              impactD2 = d2;
              impactIndex = candidateIndex;
            }
          }
        }
      }
      const target = impactIndex >= 0 ? this.units[impactIndex] : null;
      if (target && this.rand() < 0.68) {
        target.hp -= projectile.damage;
        if (target.team !== projectile.team) this.metrics[projectile.team].hits++;
        if (target.hp <= 0) this.kill(target, projectile.team);
      }
      this.projectiles.splice(i, 1);
    }
  }

  step() {
    if (this.ended) return this.regimentState().alive;
    this.time += DT;
    const grid = this.buildGrid();
    const { state, teamIndices, regimentIndices, alive } = this.regimentState();
    this.updateCommanders(state, alive);
    const frontX = state.map((army, team) => {
      let weightedX = 0;
      let count = 0;
      for (let regiment = 0; regiment < 3; regiment++) {
        weightedX += army[regiment].x * army[regiment].count;
        count += army[regiment].count;
      }
      return count ? weightedX / count : teamDefenseX(team);
    });

    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      if (!u.alive) {
        u.fall = Math.min(1, u.fall + DT * 2.4);
        continue;
      }
      u.cooldown -= DT;
      const gx = Math.max(0, Math.min(GRID_W - 1, ((u.x + WORLD_W / 2) / CELL) | 0));
      const gz = Math.max(0, Math.min(GRID_D - 1, ((u.z + WORLD_D / 2) / CELL) | 0));
      let nearestEnemy = -1;
      let nearestD2 = Infinity;
      let sepX = 0;
      let sepZ = 0;
      let friends = 0;
      let pressure = 0;
      let rearSupport = 0;
      let flankThreat = 0;
      let nearbyPanic = 0;
      let bodyX = 0;
      let bodyZ = 0;

      for (let oz = -1; oz <= 1; oz++) {
        const cz = gz + oz;
        if (cz < 0 || cz >= GRID_D) continue;
        for (let ox = -1; ox <= 1; ox++) {
          const cx = gx + ox;
          if (cx < 0 || cx >= GRID_W) continue;
          const bucket = grid[cz * GRID_W + cx];
          if (!bucket) continue;
          for (const otherIndex of bucket) {
            if (otherIndex === i) continue;
            const other = this.units[otherIndex];
            const dx = other.x - u.x;
            const dz = other.z - u.z;
            const d2 = dx * dx + dz * dz;
            if (other.team !== u.team) {
              if (d2 < nearestD2) {
                nearestD2 = d2;
                nearestEnemy = otherIndex;
              }
              if (d2 < 240) pressure++;
              if (d2 < 900 && d2 > 0.01) {
                const d = Math.sqrt(d2);
                const forwardDot = (dx * Math.cos(u.facing) + dz * Math.sin(u.facing)) / d;
                if (forwardDot < -0.2) flankThreat++;
              }
              const collisionRange = UNIT_RADIUS * 2;
              if (d2 < collisionRange * collisionRange && d2 > 0.01) {
                const d = Math.sqrt(d2);
                const force = (collisionRange - d) / collisionRange;
                bodyX -= dx / d * force;
                bodyZ -= dz / d * force;
              }
            } else if (d2 < 625 && d2 > 0.01) {
              const d = Math.sqrt(d2);
              if (d2 < 110) {
                const force = (10.5 - d) / 10.5;
                sepX -= dx / d * force;
                sepZ -= dz / d * force;
              }
              friends++;
              const forwardDot = (dx * Math.cos(u.facing) + dz * Math.sin(u.facing)) / d;
              const aligned = Math.cos(angleDelta(u.facing, other.facing)) > 0.72;
              if (forwardDot < -0.25 && aligned) rearSupport++;
              if (other.routing || other.morale < 0.18) nearbyPanic++;
            }
          }
        }
      }

      const dir = u.team === 0 ? 1 : -1;
      const plan = this.plans[u.team][u.regiment];
      const own = state[u.team][u.regiment];
      const enemyArmy = state[u.team === 0 ? 1 : 0];
      let targetState = enemyArmy[plan.targetRegiment];
      if (!targetState.count) {
        let best = Infinity;
        for (const candidate of enemyArmy) {
          if (!candidate.count) continue;
          const distance = Math.hypot(candidate.x - u.x, candidate.z - u.z);
          if (distance < best) {
            best = distance;
            targetState = candidate;
          }
        }
      }

      const endurance = 1 - Math.min(0.3, u.fatigue * 0.3);
      const marchSpeed = (u.type ? 21 : 28) * plan.pace * endurance;
      let targetVx = 0;
      let targetVz = 0;
      const desiredX = own.x + u.formationOffsetX - plan.formationCenterX;
      const desiredZ = own.z + u.formationOffsetZ;
      let goalX = null;
      let goalZ = 0;
      let goalScale = 1;

      if (plan.mission === 'crowd' && targetState?.count) {
        const dx = targetState.x - own.x;
        const dz = targetState.z - own.z;
        const distance = Math.hypot(dx, dz) || 1;
        targetVx = dx / distance * marchSpeed;
        targetVz = dz / distance * marchSpeed;
      } else if (plan.mission === 'neural_hold') {
        goalX = plan.role === 'archer'
          ? frontX[u.team] - dir * 125
          : teamDefenseX(u.team) - (plan.role === 'reserve' ? dir * 80 : 0);
        goalZ = sectorCenter(plan.targetSector ?? 3);
        goalScale = 0.82;
      } else if (plan.mission === 'neural_reserve') {
        goalX = teamDefenseX(u.team) - dir * 155;
        goalZ = sectorCenter(plan.targetSector ?? 3);
        goalScale = 0.76;
      } else if (plan.mission === 'neural_advance' && targetState?.count) {
        goalX = plan.role === 'archer'
          ? this.commanders[u.team].pursuit
            ? targetState.x - dir * 260
            : frontX[u.team] - dir * 105
          : targetState.x - dir * 30;
        goalZ = sectorCenter(plan.targetSector ?? 3);
        goalScale = 0.92;
      } else if (plan.mission === 'neural_flank' && targetState?.count) {
        const flankSign = (plan.targetSector ?? 3) < 3 ? -1 : 1;
        const outerZ = flankSign * 395;
        if (Math.abs(own.z - outerZ) > 50) {
          goalX = -dir * 75;
          goalZ = outerZ;
        } else {
          goalX = targetState.x + dir * 75;
          goalZ = targetState.z + flankSign * 70;
        }
        goalScale = 0.98;
      } else if (plan.mission === 'neural_assault' && targetState?.count) {
        const dx = targetState.x - own.x;
        const dz = sectorCenter(plan.targetSector ?? 3) - own.z;
        const distance = Math.hypot(dx, dz) || 1;
        targetVx = dx / distance * marchSpeed;
        targetVz = dz / distance * marchSpeed;
      } else if (plan.mission === 'screen') {
        goalX = teamDefenseX(u.team);
        goalZ = [-135, 0, 135][u.regiment] ?? plan.homeZ;
        goalScale = 0.85;
      } else if (plan.mission === 'wide_screen') {
        goalX = teamDefenseX(u.team);
        goalZ = [-245, 0, 245][u.regiment] ?? plan.homeZ;
        goalScale = 0.88;
      } else if (plan.mission === 'reserve') {
        goalX = teamDefenseX(u.team) - dir * 145;
        goalZ = plan.homeZ * 0.35;
        goalScale = 0.8;
      } else if (plan.mission === 'guard') {
        goalX = teamDefenseX(u.team) - dir * 45;
        goalZ = plan.homeZ;
        goalScale = 0.9;
      } else if (plan.mission === 'wide_guard') {
        goalX = teamDefenseX(u.team) - dir * 30;
        goalZ = plan.homeZ < 0 ? -360 : 360;
        goalScale = 0.92;
      } else if ((plan.mission === 'fire_support' || plan.mission === 'defensive_fire') && targetState?.count) {
        goalX = frontX[u.team] - dir * (plan.mission === 'defensive_fire' ? 125 : 105);
        goalZ = targetState.z;
        goalScale = plan.mission === 'defensive_fire' ? 0.78 : 0.9;
      } else if (plan.mission === 'flank' && targetState?.count) {
        const outerZ = plan.flankSign * 385;
        if (Math.abs(own.z - outerZ) > 55) {
          goalX = -dir * 60;
          goalZ = outerZ;
        } else {
          goalX = targetState.x + dir * 70;
          goalZ = targetState.z + plan.flankSign * 75;
        }
      } else if (targetState?.count) {
        const dx = targetState.x - own.x;
        const dz = targetState.z - own.z;
        const distance = Math.hypot(dx, dz) || 1;
        targetVx = dx / distance * marchSpeed;
        const convergence = plan.mission === 'assault' || plan.mission === 'counterattack' ? 1 : 0.65;
        targetVz = dz / distance * marchSpeed * convergence;
      }

      if (goalX !== null) {
        const dx = goalX - own.x;
        const dz = goalZ - own.z;
        const distance = Math.hypot(dx, dz);
        if (distance >= 8) {
          const speed = Math.min(marchSpeed * goalScale, distance * 0.7);
          targetVx = dx / distance * speed;
          targetVz = dz / distance * speed;
        }
      }
      if (plan.mission !== 'crowd') {
        const cohesion = nearestD2 < 2500 ? 0 : 0.32;
        targetVx += (desiredX - u.x) * cohesion;
        targetVz += (desiredZ - u.z) * cohesion;
      }

      if (u.type === 1) {
        const enemyTeam = u.team === 0 ? 1 : 0;
        const commander = this.commanders[u.team];
        const orderedTargets = commander.mode === 'crowd'
          ? null
          : regimentIndices[enemyTeam][plan.targetRegiment];
        let rangedIndex = -1;
        let rangedD2 = 340 * 340;
        const targetPools = orderedTargets?.length
          ? [orderedTargets, teamIndices[enemyTeam]]
          : [teamIndices[enemyTeam]];
        for (const candidates of targetPools) {
          const samples = Math.min(14, candidates.length);
          for (let sample = 0; sample < samples; sample++) {
            const candidate = candidates[(this.rand() * candidates.length) | 0];
            const enemy = this.units[candidate];
            const dx = enemy.x - u.x;
            const dz = enemy.z - u.z;
            const d2 = dx * dx + dz * dz;
            if (d2 < rangedD2) {
              rangedD2 = d2;
              rangedIndex = candidate;
            }
          }
          if (rangedIndex >= 0) break;
        }
        if (rangedIndex >= 0) {
          const target = this.units[rangedIndex];
          const distance = Math.sqrt(rangedD2);
          targetVx = distance < 90 ? -dir * 36 : targetVx * 0.16;
          const fireMode = commander.mode === 'crowd'
            ? 'independent'
            : plan.fireMode ?? 'independent';
          const freeFire = fireMode === 'independent' && u.cooldown <= 0;
          const volley = fireMode === 'volley'
            && this.time <= commander.volleyUntil
            && this.time >= commander.volleyStart + u.volleyDelay
            && u.lastVolley !== commander.volleyId
            && u.cooldown <= 0;
          if ((freeFire || volley) && distance > 25) {
            this.fireArrow(u, rangedIndex, target);
            u.lastVolley = commander.volleyId;
            u.cooldown = SHOT_CADENCE;
          }
        }
      }

      if (nearestEnemy >= 0) {
        const enemy = this.units[nearestEnemy];
        const dx = enemy.x - u.x;
        const dz = enemy.z - u.z;
        const distance = Math.sqrt(nearestD2) || 1;
        const enemyHeading = Math.atan2(dz, dx);
        const turnRate = (u.type ? 2.7 : 3.25) * (1 - u.fatigue * 0.25);
        u.facing = turnTowards(u.facing, enemyHeading, turnRate * DT);
        const facingDot = Math.cos(angleDelta(u.facing, enemyHeading));
        if (distance < 11) {
          targetVx = -dx / distance * 3;
          targetVz = -dz / distance * 3;
          if (u.cooldown <= 0 && facingDot >= ATTACK_ARC_COS) {
            enemy.hp -= u.type ? 3 + this.rand() * 5 : 7 + this.rand() * 8;
            u.cooldown = (0.58 + this.rand() * 0.55) * (1 + u.fatigue * 0.45);
            u.fatigue = Math.min(1, u.fatigue + (u.type ? 0.008 : 0.014));
            this.metrics[u.team].meleeHits++;
            if (enemy.hp <= 0) this.kill(enemy, u.team);
          }
        } else if (distance < 45) {
          targetVx = dx / distance * 39;
          targetVz = dz / distance * 39;
        }
      }

      const localPressure = pressure + flankThreat * 0.8 + nearbyPanic * 0.55
        - friends * 0.16 - rearSupport * 0.22;
      if (localPressure > 1.5) {
        u.morale -= DT * (0.012 * localPressure + u.fatigue * 0.006);
      } else {
        u.morale = Math.min(1, u.morale + DT * (pressure ? 0.001 : 0.005));
      }
      if (u.morale < 0.18) u.routing = true;
      else if (u.routing && u.morale > 0.42 && pressure === 0) u.routing = false;
      if (u.routing) {
        if (nearestEnemy >= 0) {
          const enemy = this.units[nearestEnemy];
          const dx = enemy.x - u.x;
          const dz = enemy.z - u.z;
          const distance = Math.hypot(dx, dz) || 1;
          targetVx = -dx / distance * 50 - dir * 14;
          targetVz = -dz / distance * 50 + (this.rand() - 0.5) * 8;
        } else {
          targetVx = -dir * 46;
        }
      }

      targetVx += sepX * 23;
      targetVz += sepZ * 23;
      targetVx += bodyX * 48;
      targetVz += bodyZ * 48;
      const response = Math.min(1, DT * 5.5);
      u.vx += (targetVx - u.vx) * response;
      u.vz += (targetVz - u.vz) * response;
      const speed = Math.hypot(u.vx, u.vz);
      if (speed > 54) {
        u.vx *= 54 / speed;
        u.vz *= 54 / speed;
      }
      if (nearestD2 >= 2025 && speed > 2) {
        const movementHeading = Math.atan2(u.vz, u.vx);
        const turnRate = (u.type ? 2.7 : 3.25) * (1 - u.fatigue * 0.25);
        u.facing = turnTowards(u.facing, movementHeading, turnRate * DT);
      }
      if (speed > 18) u.fatigue = Math.min(1, u.fatigue + DT * 0.0022 * (speed / 28));
      else if (speed < 7 && pressure === 0) u.fatigue = Math.max(0, u.fatigue - DT * 0.006);
      u.x = Math.max(-WORLD_W / 2 + 8, Math.min(WORLD_W / 2 - 8, u.x + u.vx * DT));
      u.z = Math.max(-WORLD_D / 2 + 8, Math.min(WORLD_D / 2 - 8, u.z + u.vz * DT));
      if (u.routing && (
        Math.abs(u.x) >= WORLD_W / 2 - 9
        || Math.abs(u.z) >= WORLD_D / 2 - 9
      )) {
        // A routed soldier who reaches a field edge has escaped the battle.
        // This is a withdrawal, not a kill, so the opponent receives no kill credit.
        u.alive = false;
        u.fall = 1;
      }
    }

    this.updateProjectiles(grid);
    const finalAlive = this.regimentState().alive;
    if (finalAlive[0] === 0 || finalAlive[1] === 0 || this.time >= MAX_TIME) this.ended = true;
    return finalAlive;
  }

  setFireMode(team, mode) {
    if (!['auto', ...V4_FIRE_MODES].includes(mode)) return;
    const commander = this.commanders[team];
    commander.fireModeOverride = mode;
    if (commander.mode === 'crowd') {
      commander.fireModeOverride = 'auto';
      mode = 'independent';
    }
    if (mode !== 'auto') {
      for (const plan of this.plans[team]) {
        if (plan.role === 'archer') plan.fireMode = mode;
      }
    }
  }

  setStrategy(team, mode) {
    const commander = this.commanders[team];
    if (!commander || !STRATEGIES.includes(mode)) return;
    commander.mode = mode;
    commander.policy = mode === 'ppo'
      ? (isCompatiblePpoPolicy(TRAINED_PPO_POLICY) ? TRAINED_PPO_POLICY : createPpoPolicy())
      : mode === 'commander_v3'
        ? (isCompatibleV3Policy(TRAINED_COMMANDER_V3_POLICY)
          ? TRAINED_COMMANDER_V3_POLICY
          : createV3Policy())
        : mode === 'commander_v4'
          ? (isCompatibleV4Policy(TRAINED_COMMANDER_V4_POLICY)
            ? TRAINED_COMMANDER_V4_POLICY
            : createV4Policy())
        : TRAINED_POLICY;
    commander.recurrentState = Array.from(
      { length: REGIMENTS },
      () => new Array(PPO_MEMORY).fill(0)
    );
    commander.v3Hidden = new Array(V3_MEMORY).fill(0);
    commander.v4Hidden = new Array(V4_MEMORY).fill(0);
    commander.nextDecision = this.time;
    commander.nextVolley = this.time + SHOT_CADENCE;
    commander.volleyStart = -1;
    commander.volleyUntil = -1;
    for (const plan of this.plans[team]) {
      plan.mission = mode === 'crowd' ? 'crowd' : 'advance';
      plan.fireMode = 'independent';
    }
  }

  run() {
    let alive = [this.unitsPerArmy, this.unitsPerArmy];
    while (this.time < MAX_TIME && alive[0] > 0 && alive[1] > 0) {
      alive = this.step();
    }
    const final = this.regimentState().alive;
    const winner = final[0] === final[1] ? 'draw' : final[0] > final[1] ? 'blue' : 'red';
    const rewards = [0, 0];
    if (winner === 'blue') rewards[0] = 1, rewards[1] = -1;
    else if (winner === 'red') rewards[0] = -1, rewards[1] = 1;
    const survivorEdge = (final[0] - final[1]) / Math.max(1, this.unitsPerArmy);
    rewards[0] += survivorEdge * 0.4;
    rewards[1] -= survivorEdge * 0.4;
    return {
      winner,
      duration: this.time,
      survivors: final,
      metrics: this.metrics,
      commanderDecisions: this.commanders.map(commander => commander.decisions),
      trainingSamples: this.trainingRecords.map((records, team) =>
        records.some(record =>
          record.kind === 'commander_v3' || record.kind === 'commander_v4'
        )
          ? finalizeV3Records(records, rewards[team])
          : records.some(record => record.kind === 'ppo')
            ? finalizePpoRecords(records, rewards[team])
          : records.map(record => ({ ...record, reward: rewards[team] }))
      )
    };
  }
}

function runMatchup({
  blue, red, units, trials, seed, blueFire = 'auto', redFire = 'auto'
}) {
  const summary = {
    blue,
    red,
    trials,
    blueWins: 0,
    redWins: 0,
    draws: 0,
    blueSurvivors: 0,
    redSurvivors: 0,
    duration: 0,
    blueArrows: 0,
    redArrows: 0,
    blueHits: 0,
    redHits: 0,
    blueMelee: 0,
    redMelee: 0
  };
  for (let trial = 0; trial < trials; trial++) {
    const battle = new Battle({
      blue,
      red,
      units,
      seed: seed + trial * 7919,
      fireModes: [blueFire, redFire]
    });
    const result = battle.run();
    if (result.winner === 'blue') summary.blueWins++;
    else if (result.winner === 'red') summary.redWins++;
    else summary.draws++;
    summary.blueSurvivors += result.survivors[0];
    summary.redSurvivors += result.survivors[1];
    summary.duration += result.duration;
    summary.blueArrows += result.metrics[0].arrows;
    summary.redArrows += result.metrics[1].arrows;
    summary.blueHits += result.metrics[0].hits;
    summary.redHits += result.metrics[1].hits;
    summary.blueMelee += result.metrics[0].meleeHits;
    summary.redMelee += result.metrics[1].meleeHits;
  }
  for (const key of [
    'blueSurvivors', 'redSurvivors', 'duration', 'blueArrows',
    'redArrows', 'blueHits', 'redHits', 'blueMelee', 'redMelee'
  ]) {
    summary[key] /= trials;
  }
  return summary;
}

function printable(summary) {
  return {
    blue: STRATEGY_NAMES[summary.blue],
    red: STRATEGY_NAMES[summary.red],
    score: `${summary.blueWins}-${summary.redWins}-${summary.draws}`,
    survivors: `${summary.blueSurvivors.toFixed(1)}:${summary.redSurvivors.toFixed(1)}`,
    seconds: summary.duration.toFixed(1),
    arrows: `${summary.blueArrows.toFixed(0)}:${summary.redArrows.toFixed(0)}`,
    hits: `${summary.blueHits.toFixed(0)}:${summary.redHits.toFixed(0)}`,
    melee: `${summary.blueMelee.toFixed(0)}:${summary.redMelee.toFixed(0)}`
  };
}

function combineSummaries(parts) {
  const totalTrials = parts.reduce((sum, part) => sum + part.trials, 0);
  const combined = {
    blue: parts[0].blue,
    red: parts[0].red,
    trials: totalTrials,
    blueWins: 0,
    redWins: 0,
    draws: 0,
    blueSurvivors: 0,
    redSurvivors: 0,
    duration: 0,
    blueArrows: 0,
    redArrows: 0,
    blueHits: 0,
    redHits: 0,
    blueMelee: 0,
    redMelee: 0
  };
  for (const part of parts) {
    combined.blueWins += part.blueWins;
    combined.redWins += part.redWins;
    combined.draws += part.draws;
    for (const key of [
      'blueSurvivors', 'redSurvivors', 'duration', 'blueArrows',
      'redArrows', 'blueHits', 'redHits', 'blueMelee', 'redMelee'
    ]) {
      combined[key] += part[key] * part.trials;
    }
  }
  for (const key of [
    'blueSurvivors', 'redSurvivors', 'duration', 'blueArrows',
    'redArrows', 'blueHits', 'redHits', 'blueMelee', 'redMelee'
  ]) {
    combined[key] /= totalTrials;
  }
  return combined;
}

class SimulationPool {
  constructor(size) {
    this.queue = [];
    this.slots = Array.from({ length: size }, () => {
      const worker = new WorkerCtor(new URL(import.meta.url), { workerData: { poolWorker: true } });
      const slot = { worker, current: null };
      worker.on('message', result => {
        const current = slot.current;
        slot.current = null;
        current?.resolve(result);
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

async function runMatchupParallel(options, pool) {
  const workerCount = Math.min(Math.floor(options.workers), options.trials);
  if (workerCount <= 1 || !pool) return runMatchup(options);
  const jobs = [];
  let trialOffset = 0;
  for (let workerIndex = 0; workerIndex < workerCount; workerIndex++) {
    const trials = Math.floor(options.trials / workerCount) + (workerIndex < options.trials % workerCount ? 1 : 0);
    const job = {
      blue: options.blue,
      red: options.red,
      units: options.units,
      trials,
      seed: options.seed + trialOffset * 7919
      ,
      blueFire: options.blueFire,
      redFire: options.redFire
    };
    trialOffset += trials;
    jobs.push(pool.run(job));
  }
  return combineSummaries(await Promise.all(jobs));
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const started = performance.now();
  const results = [];
  const poolSize = Math.min(Math.floor(options.workers), options.trials);
  const pool = poolSize > 1 ? new SimulationPool(poolSize) : null;

  try {
    if (options.matrix) {
      for (const blue of STRATEGIES) {
        for (const red of STRATEGIES) {
          results.push(await runMatchupParallel({ ...options, blue, red }, pool));
        }
      }
    } else {
      results.push(await runMatchupParallel(options, pool));
    }
  } finally {
    await pool?.close();
  }

  const elapsed = (performance.now() - started) / 1000;
  if (options.json) {
    console.log(JSON.stringify({ options, elapsed, results }, null, 2));
  } else {
    console.table(results.map(printable));
    console.log(`Прогон: ${results.length * options.trials} боёв за ${elapsed.toFixed(2)} с на ${Math.min(options.workers, options.trials)} worker(ах)`);
    console.log('score = победы синих-победы красных-ничьи; остальные показатели — средние за бой');
  }
}

if (isNodeRuntime) {
  const os = await import('node:os');
  const { pathToFileURL } = await import('node:url');
  const workerThreads = await import('node:worker_threads');
  cpuParallelism = os.availableParallelism();
  WorkerCtor = workerThreads.Worker;
  nodeIsMainThread = workerThreads.isMainThread;
  nodeParentPort = workerThreads.parentPort;
  nodeWorkerData = workerThreads.workerData;
  const directRun = process.argv[1]
    ? import.meta.url === pathToFileURL(process.argv[1]).href
    : false;
  if (nodeIsMainThread && directRun) {
    await main();
  } else if (nodeWorkerData?.poolWorker) {
    nodeParentPort.on('message', job => {
      nodeParentPort.postMessage(runMatchup(job));
    });
  }
}
