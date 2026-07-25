import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Battle as BattleCore } from './diagnostic.js';

const WORLD_W = 1600;
const WORLD_D = 900;
const REGIMENTS = 6;
const ARCHER_RATIO = 0.22;
const SHOT_CADENCE = 2.1;
const REGIMENT_ROLES = ['line', 'line', 'line', 'archer', 'reserve', 'flank'];
const REGIMENT_WEIGHTS = [0.16, 0.16, 0.16, ARCHER_RATIO, 0.16, 0.14];
const CELL = 30;
const GRID_W = Math.ceil(WORLD_W / CELL);
const GRID_D = Math.ceil(WORLD_D / CELL);
const MAX_PER_ARMY = 1200;
const MAX_UNITS = MAX_PER_ARMY * 2;
const BLUE = 0x2296f3;
const BLUE_DARK = 0x0b5ca8;
const RED = 0xff5d40;
const RED_DARK = 0xa92e1e;

const canvas = document.querySelector('#battlefield');
const unitsInput = document.querySelector('#units');
const speedInput = document.querySelector('#speed');
const blueStrategyInput = document.querySelector('#blue-strategy');
const redStrategyInput = document.querySelector('#red-strategy');
const unitsValue = document.querySelector('#units-value');
const speedValue = document.querySelector('#speed-value');
const pauseButton = document.querySelector('#pause');
const resetButton = document.querySelector('#reset');
const cameraResetButton = document.querySelector('#camera-reset');
const blueCountEl = document.querySelector('#blue-count');
const redCountEl = document.querySelector('#red-count');
const blueOrderEl = document.querySelector('#blue-order');
const redOrderEl = document.querySelector('#red-order');
const phaseEl = document.querySelector('#phase');
const battleTimeEl = document.querySelector('#battle-time');
const fpsEl = document.querySelector('#fps');
const loading = document.querySelector('#loading');

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: 'high-performance'
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101419);
scene.fog = new THREE.FogExp2(0x101419, 0.0009);

const camera = new THREE.PerspectiveCamera(46, window.innerWidth / window.innerHeight, 1, 5000);
camera.position.set(0, 650, 850);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.075;
controls.minDistance = 80;
controls.maxDistance = 2300;
controls.maxPolarAngle = Math.PI * 0.49;
controls.screenSpacePanning = false;

scene.add(new THREE.HemisphereLight(0xbcd9ef, 0x24301f, 2.1));
const sun = new THREE.DirectionalLight(0xfff1d6, 2.5);
sun.position.set(-500, 900, 300);
scene.add(sun);

const groundMaterial = new THREE.MeshStandardMaterial({
  color: 0x263127,
  roughness: 1,
  metalness: 0
});
const ground = new THREE.Mesh(new THREE.PlaneGeometry(WORLD_W, WORLD_D, 1, 1), groundMaterial);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.2;
scene.add(ground);

const grid = new THREE.GridHelper(WORLD_W, 40, 0x435047, 0x313b34);
grid.position.y = 0;
grid.material.transparent = true;
grid.material.opacity = 0.34;
scene.add(grid);

const fieldEdge = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.PlaneGeometry(WORLD_W, WORLD_D)),
  new THREE.LineBasicMaterial({ color: 0x657066, transparent: true, opacity: 0.35 })
);
fieldEdge.rotation.x = -Math.PI / 2;
fieldEdge.position.y = 0.08;
scene.add(fieldEdge);

const infantryGeometry = new THREE.CapsuleGeometry(3.1, 7.5, 2, 5);
infantryGeometry.translate(0, 6.8, 0);
const archerGeometry = new THREE.ConeGeometry(3.7, 11, 5);
archerGeometry.translate(0, 5.5, 0);

function makeMaterial(color) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.72,
    metalness: 0.04
  });
}

const unitMeshes = {
  blueInfantry: new THREE.InstancedMesh(infantryGeometry, makeMaterial(BLUE), MAX_PER_ARMY),
  blueArchers: new THREE.InstancedMesh(archerGeometry, makeMaterial(BLUE_DARK), MAX_PER_ARMY),
  redInfantry: new THREE.InstancedMesh(infantryGeometry, makeMaterial(RED), MAX_PER_ARMY),
  redArchers: new THREE.InstancedMesh(archerGeometry, makeMaterial(RED_DARK), MAX_PER_ARMY)
};

for (const mesh of Object.values(unitMeshes)) {
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.frustumCulled = false;
  mesh.count = 0;
  scene.add(mesh);
}

const flagGroup = new THREE.Group();
scene.add(flagGroup);
const flagPoles = [];
for (let team = 0; team < 2; team++) {
  for (let regiment = 0; regiment < REGIMENTS; regiment++) {
    const color = team === 0 ? BLUE : RED;
    const pole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.7, 0.7, 28, 5),
      new THREE.MeshStandardMaterial({ color: 0xd7d9d2, roughness: 0.6 })
    );
    pole.position.y = 14;
    const flag = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 10),
      new THREE.MeshStandardMaterial({ color, side: THREE.DoubleSide, roughness: 0.7 })
    );
    flag.position.set(team === 0 ? 10 : -10, 23, 0);
    const holder = new THREE.Group();
    holder.add(pole, flag);
    holder.visible = false;
    flagGroup.add(holder);
    flagPoles.push({ team, regiment, holder });
  }
}

const MAX_ARROWS = 3500;
const arrowPositions = new Float32Array(MAX_ARROWS * 6);
const arrowGeometry = new THREE.BufferGeometry();
arrowGeometry.setAttribute('position', new THREE.BufferAttribute(arrowPositions, 3));
arrowGeometry.setDrawRange(0, 0);
const arrowLines = new THREE.LineSegments(
  arrowGeometry,
  new THREE.LineBasicMaterial({ color: 0xe8d8b5, transparent: true, opacity: 0.88 })
);
arrowLines.frustumCulled = false;
scene.add(arrowLines);

const dummy = new THREE.Object3D();
let units = [];
let regimentPlans = [[], []];
let armyCommanders = [];
let coreBattle = null;
let unitCount = 0;
let arrows = [];
let simTime = 0;
let paused = false;
let ended = false;
let accumulator = 0;
let lastFrame = performance.now();
let fpsTimer = 0;
let fpsFrames = 0;
let shownFps = 60;
let seed = 1;
const keys = new Set();

function rand() {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
}

function createUnit(team, type, regiment, x, z, formationOffsetX, formationOffsetZ) {
  const maxHp = type === 1 ? 38 : 60;
  return {
    team,
    type,
    regiment,
    x,
    z,
    formationOffsetX,
    formationOffsetZ,
    vx: (team === 0 ? 1 : -1) * (12 + rand() * 4),
    vz: (rand() - 0.5) * 2,
    hp: maxHp * (0.78 + rand() * 0.22),
    maxHp,
    cooldown: rand() * (type === 1 ? SHOT_CADENCE : 0.8),
    morale: 0.84 + rand() * 0.16,
    alive: true,
    fall: 0,
    lastVolley: -1,
    phase: rand() * Math.PI * 2
  };
}

function shuffledRegiments() {
  const values = Array.from({ length: REGIMENTS }, (_, index) => index);
  for (let i = values.length - 1; i > 0; i--) {
    const j = (rand() * (i + 1)) | 0;
    [values[i], values[j]] = [values[j], values[i]];
  }
  return values;
}

function createArmyDisposition(team) {
  const plans = [];
  for (let regiment = 0; regiment < REGIMENTS; regiment++) {
    const role = REGIMENT_ROLES[regiment];
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
      order: 'advance',
      mission: 'advance',
      flankSign: rand() < 0.5 ? -1 : 1,
      formationCenterX: 0
    });
  }
  return plans;
}

function createCommander(team, mode) {
  return {
    team,
    mode,
    nextDecision: 0,
    nextVolley: SHOT_CADENCE,
    volleyUntil: -1,
    volleyId: 0,
    tactic: 'advance',
    label: 'ОБЩЕЕ НАСТУПЛЕНИЕ'
  };
}

function clearBattle() {
  for (let i = 0; i < unitCount; i++) units[i] = undefined;
  unitCount = 0;
  arrows = [];
}

function resetBattle() {
  const battleSeed = (Date.now() ^ (Math.random() * 0xffffffff)) >>> 0;
  coreBattle = new BattleCore({
    blue: blueStrategyInput.value,
    red: redStrategyInput.value,
    units: Number(unitsInput.value),
    seed: battleSeed
  });
  units = coreBattle.units;
  unitCount = units.length;
  arrows = coreBattle.projectiles;
  regimentPlans = coreBattle.plans;
  armyCommanders = coreBattle.commanders;
  simTime = coreBattle.time;
  accumulator = 0;
  ended = false;
  paused = false;
  pauseButton.textContent = 'Пауза';
  updateMeshes();
  updateHud();
  return;

  clearBattle();
  seed = (Date.now() ^ (Math.random() * 0xffffffff)) >>> 0;
  const count = Number(unitsInput.value);
  regimentPlans = [createArmyDisposition(0), createArmyDisposition(1)];
  armyCommanders = [
    createCommander(0, blueStrategyInput.value),
    createCommander(1, redStrategyInput.value)
  ];
  const blueTargets = shuffledRegiments();
  const redTargets = shuffledRegiments();
  for (let regiment = 0; regiment < REGIMENTS; regiment++) {
    regimentPlans[0][regiment].targetRegiment = blueTargets[regiment];
    regimentPlans[1][regiment].targetRegiment = redTargets[regiment];
  }
  const archerTotal = Math.round(count * ARCHER_RATIO);
  const infantryTotal = count - archerTotal;
  const infantryWeights = [0.205, 0.205, 0.205, 0, 0.205, 0.18];
  const regimentCounts = new Array(REGIMENTS).fill(0);
  regimentCounts[3] = archerTotal;
  let infantryAssigned = 0;
  const infantryRegiments = [0, 1, 2, 4, 5];
  for (let index = 0; index < infantryRegiments.length; index++) {
    const regiment = infantryRegiments[index];
    const isLast = index === infantryRegiments.length - 1;
    const regimentCount = isLast
      ? infantryTotal - infantryAssigned
      : Math.floor(infantryTotal * infantryWeights[regiment]);
    regimentCounts[regiment] = regimentCount;
    infantryAssigned += regimentCount;
  }

  for (let team = 0; team < 2; team++) {
    const dir = team === 0 ? 1 : -1;
    let made = 0;
    for (let regiment = 0; regiment < REGIMENTS; regiment++) {
      const regimentCount = regimentCounts[regiment];
      const rows = Math.max(6, Math.ceil(Math.sqrt(regimentCount * 1.15)));
      const cols = Math.ceil(regimentCount / rows);
      const plan = regimentPlans[team][regiment];
      const laneZ = plan.z;
      const baseX = plan.x;
      let offsetSumX = 0;
      let spawnedInRegiment = 0;

      for (let n = 0; n < regimentCount && made < count; n++) {
        const type = plan.role === 'archer' ? 1 : 0;
        const row = n % rows;
        const col = Math.floor(n / rows);
        const centeredRow = row - (rows - 1) / 2;
        const formationOffsetX = -dir * col * 10;
        const x = baseX + formationOffsetX + (rand() - 0.5) * 2;
        const formationOffsetZ = centeredRow * 10;
        const z = laneZ + formationOffsetZ + (rand() - 0.5) * 2;
        units[unitCount++] = createUnit(team, type, regiment, x, z, formationOffsetX, formationOffsetZ);
        offsetSumX += formationOffsetX;
        spawnedInRegiment++;
        made++;
      }
      plan.formationCenterX = spawnedInRegiment ? offsetSumX / spawnedInRegiment : 0;
      plan.initialCount = regimentCount;
    }
  }

  simTime = 0;
  accumulator = 0;
  ended = false;
  paused = false;
  pauseButton.textContent = 'Пауза';
  updateMeshes();
  updateHud();
}

function buildSpatialGrid() {
  const cells = new Array(GRID_W * GRID_D);
  for (let i = 0; i < unitCount; i++) {
    const u = units[i];
    if (!u.alive) continue;
    const gx = Math.max(0, Math.min(GRID_W - 1, ((u.x + WORLD_W / 2) / CELL) | 0));
    const gz = Math.max(0, Math.min(GRID_D - 1, ((u.z + WORLD_D / 2) / CELL) | 0));
    const index = gz * GRID_W + gx;
    if (!cells[index]) cells[index] = [];
    cells[index].push(i);
  }
  return cells;
}

function killUnit(u) {
  u.alive = false;
  u.fall = 0.01;
  u.vx = (rand() - 0.5) * 8;
  u.vz = (rand() - 0.5) * 8;
}

function fireArrow(shooter, targetIndex, target) {
  if (arrows.length >= MAX_ARROWS) return;
  const dx = target.x - shooter.x;
  const dz = target.z - shooter.z;
  const distance = Math.hypot(dx, dz);
  const flight = distance / 430;
  arrows.push({
    x: shooter.x,
    y: 11,
    z: shooter.z,
    vx: dx / flight,
    vy: 68 + distance * 0.025,
    vz: dz / flight,
    life: flight,
    target: targetIndex,
    damage: 8 + rand() * 8
  });
}

function issueCommanderOrders(team, ownState, enemyState, alive, initialCount) {
  const commander = armyCommanders[team];
  if (simTime < commander.nextDecision) return;
  commander.nextDecision = simTime + 8;

  const livingEnemies = enemyState
    .map((state, regiment) => ({ ...state, regiment }))
    .filter(state => state.count > 0);
  if (!livingEnemies.length) return;

  if (commander.mode === 'crowd') {
    commander.tactic = 'crowd';
    commander.label = 'БЕЗ КОМАНДОВАНИЯ';
    for (let regiment = 0; regiment < REGIMENTS; regiment++) {
      const own = ownState[regiment];
      if (!own.count) continue;
      const plan = regimentPlans[team][regiment];
      let nearest = livingEnemies[0];
      let nearestDistance = Infinity;
      for (const enemy of livingEnemies) {
        const distance = Math.hypot(enemy.x - own.x, enemy.z - own.z);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = enemy;
        }
      }
      plan.mission = 'crowd';
      plan.order = 'crowd';
      plan.targetRegiment = nearest.regiment;
    }
    return;
  }

  const enemyAlive = livingEnemies.reduce((sum, state) => sum + state.count, 0);
  const relativeStrength = alive / Math.max(1, enemyAlive);
  let closestThreat = livingEnemies[0];
  let closestThreatDistance = Infinity;
  for (const own of ownState) {
    if (!own.count) continue;
    for (const enemy of livingEnemies) {
      const distance = Math.hypot(enemy.x - own.x, enemy.z - own.z);
      if (distance < closestThreatDistance) {
        closestThreatDistance = distance;
        closestThreat = enemy;
      }
    }
  }

  const assaultRegiments = [ownState[1], ownState[2], ownState[4]].filter(state => state.count);
  const assaultCenter = assaultRegiments.reduce(
    (center, state) => ({ x: center.x + state.x / assaultRegiments.length, z: center.z + state.z / assaultRegiments.length }),
    { x: 0, z: 0 }
  );
  const assaultTarget = livingEnemies.reduce((best, current) => {
    const bestScore = Math.hypot(best.x - assaultCenter.x, best.z - assaultCenter.z) + best.count * 2;
    const currentScore = Math.hypot(current.x - assaultCenter.x, current.z - assaultCenter.z) + current.count * 2;
    return currentScore < bestScore ? current : best;
  });
  const flankTargets = {
    '-1': livingEnemies.reduce((outer, current) => current.z < outer.z ? current : outer),
    '1': livingEnemies.reduce((outer, current) => current.z > outer.z ? current : outer)
  };
  const breachedEnemies = livingEnemies.filter(enemy =>
    team === 0 ? enemy.x < -120 : enemy.x > 120
  );

  let doctrine = commander.mode;
  if (commander.mode === 'adaptive') {
    doctrine = relativeStrength < 0.92 || (closestThreatDistance < 190 && relativeStrength < 1.08)
      ? 'defensive'
      : 'offensive';
  }
  commander.tactic = doctrine;
  commander.label = commander.mode === 'adaptive'
    ? `АДАПТИВНО: ${doctrine === 'offensive' ? 'НАСТУПЛЕНИЕ' : 'ОБОРОНА'}`
    : doctrine === 'offensive' ? 'НАСТУПЛЕНИЕ ПО РОЛЯМ' : 'ЭШЕЛОНИРОВАННАЯ ОБОРОНА';

  for (let regiment = 0; regiment < REGIMENTS; regiment++) {
    const plan = regimentPlans[team][regiment];
    const own = ownState[regiment];
    if (!own.count) continue;
    let nearest = livingEnemies[0];
    let bestDistance = Infinity;
    for (const candidate of livingEnemies) {
      const distance = Math.hypot(candidate.x - own.x, candidate.z - own.z);
      if (distance < bestDistance) {
        bestDistance = distance;
        nearest = candidate;
      }
    }

    if (doctrine === 'offensive') {
      if (plan.role === 'line') {
        plan.mission = regiment === 0 ? 'pin' : 'assault';
        plan.targetRegiment = regiment === 0 ? nearest.regiment : assaultTarget.regiment;
      } else if (plan.role === 'archer') {
        plan.mission = 'fire_support';
        plan.targetRegiment = assaultTarget.regiment;
      } else if (plan.role === 'reserve') {
        plan.mission = 'assault';
        plan.targetRegiment = assaultTarget.regiment;
      } else {
        plan.mission = 'flank';
        plan.flankSign = plan.homeZ < 0 ? -1 : 1;
        plan.targetRegiment = flankTargets[String(plan.flankSign)].regiment;
      }
    } else if (plan.role === 'line') {
      plan.mission = 'screen';
      plan.targetRegiment = nearest.regiment;
    } else if (plan.role === 'archer') {
      plan.mission = 'fire_support';
      plan.targetRegiment = closestThreat.regiment;
    } else if (plan.role === 'reserve') {
      plan.mission = breachedEnemies.length || closestThreatDistance < 260 ? 'counterattack' : 'reserve';
      plan.targetRegiment = (breachedEnemies[0] ?? closestThreat).regiment;
    } else {
      plan.mission = closestThreatDistance < 300 ? 'counterattack' : 'guard';
      plan.targetRegiment = flankTargets[String(plan.homeZ < 0 ? -1 : 1)].regiment;
    }
    plan.order = plan.mission;
  }
}

function updateCommanderAI(regimentState, aliveBlue, aliveRed) {
  const initialCount = Number(unitsInput.value);
  issueCommanderOrders(0, regimentState[0], regimentState[1], aliveBlue, initialCount);
  issueCommanderOrders(1, regimentState[1], regimentState[0], aliveRed, initialCount);

  for (const commander of armyCommanders) {
    if (commander.mode === 'crowd') continue;
    if (simTime >= commander.nextVolley) {
      commander.volleyId++;
      commander.volleyUntil = simTime + 0.75;
      commander.nextVolley += SHOT_CADENCE;
    }
  }
}

function teamDefenseX(team) {
  return team === 0 ? -185 : 185;
}

function simulate(dt) {
  if (coreBattle) {
    coreBattle.step();
    simTime = coreBattle.time;
    ended = coreBattle.ended;
    arrows = coreBattle.projectiles;
    return;
  }
  if (ended) return;
  simTime += dt;
  const cells = buildSpatialGrid();
  const teamIndices = [[], []];
  const regimentState = Array.from({ length: 2 }, () =>
    Array.from({ length: REGIMENTS }, () => ({ count: 0, x: 0, z: 0 }))
  );
  let aliveBlue = 0;
  let aliveRed = 0;

  for (let i = 0; i < unitCount; i++) {
    const u = units[i];
    if (u.alive) {
      teamIndices[u.team].push(i);
      const state = regimentState[u.team][u.regiment];
      state.count++;
      state.x += u.x;
      state.z += u.z;
      if (u.team === 0) aliveBlue++; else aliveRed++;
    }
  }
  for (const army of regimentState) {
    for (const state of army) {
      if (!state.count) continue;
      state.x /= state.count;
      state.z /= state.count;
    }
  }
  updateCommanderAI(regimentState, aliveBlue, aliveRed);

  for (let i = 0; i < unitCount; i++) {
    const u = units[i];
    if (!u.alive) {
      u.fall = Math.min(1, u.fall + dt * 2.4);
      continue;
    }

    u.cooldown -= dt;
    const gx = Math.max(0, Math.min(GRID_W - 1, ((u.x + WORLD_W / 2) / CELL) | 0));
    const gz = Math.max(0, Math.min(GRID_D - 1, ((u.z + WORLD_D / 2) / CELL) | 0));
    let nearestEnemy = -1;
    let nearestD2 = Infinity;
    let sepX = 0;
    let sepZ = 0;
    let friends = 0;
    let pressure = 0;

    for (let oz = -1; oz <= 1; oz++) {
      const cz = gz + oz;
      if (cz < 0 || cz >= GRID_D) continue;
      for (let ox = -1; ox <= 1; ox++) {
        const cx = gx + ox;
        if (cx < 0 || cx >= GRID_W) continue;
        const bucket = cells[cz * GRID_W + cx];
        if (!bucket) continue;
        for (const otherIndex of bucket) {
          if (otherIndex === i) continue;
          const other = units[otherIndex];
          const dx = other.x - u.x;
          const dz = other.z - u.z;
          const d2 = dx * dx + dz * dz;
          if (other.team !== u.team) {
            if (d2 < nearestD2) {
              nearestD2 = d2;
              nearestEnemy = otherIndex;
            }
            if (d2 < 240) pressure++;
          } else if (d2 < 110 && d2 > 0.01) {
            const d = Math.sqrt(d2);
            const force = (10.5 - d) / 10.5;
            sepX -= dx / d * force;
            sepZ -= dz / d * force;
            friends++;
          }
        }
      }
    }

    const dir = u.team === 0 ? 1 : -1;
    const plan = regimentPlans[u.team][u.regiment];
    const ownRegiment = regimentState[u.team][u.regiment];
    const enemyArmy = regimentState[u.team === 0 ? 1 : 0];
    let strategicTarget = enemyArmy[plan.targetRegiment];
    if (!strategicTarget.count) {
      let bestDistance = Infinity;
      for (const candidate of enemyArmy) {
        if (!candidate.count) continue;
        const distance = Math.hypot(candidate.x - u.x, candidate.z - u.z);
        if (distance < bestDistance) {
          bestDistance = distance;
          strategicTarget = candidate;
        }
      }
    }

    const marchSpeed = (u.type ? 21 : 28) * plan.pace;
    let targetVx = 0;
    let targetVz = 0;
    const desiredFormationX = ownRegiment.x + u.formationOffsetX - plan.formationCenterX;
    const desiredFormationZ = ownRegiment.z + u.formationOffsetZ;

    let goalX = null;
    let goalZ = 0;
    let goalSpeedScale = 1;
    if (plan.mission === 'crowd' && strategicTarget?.count) {
      const strategicDx = strategicTarget.x - u.x;
      const strategicDz = strategicTarget.z - u.z;
      const strategicDistance = Math.hypot(strategicDx, strategicDz) || 1;
      targetVx = strategicDx / strategicDistance * marchSpeed;
      targetVz = strategicDz / strategicDistance * marchSpeed;
    } else if (plan.mission === 'screen') {
      goalX = teamDefenseX(u.team);
      goalZ = [-135, 0, 135][u.regiment] ?? plan.homeZ;
      goalSpeedScale = 0.85;
    } else if (plan.mission === 'reserve') {
      goalX = teamDefenseX(u.team) - dir * 145;
      goalZ = plan.homeZ * 0.35;
      goalSpeedScale = 0.8;
    } else if (plan.mission === 'guard') {
      goalX = teamDefenseX(u.team) - dir * 45;
      goalZ = plan.homeZ;
      goalSpeedScale = 0.9;
    } else if (plan.mission === 'fire_support' && strategicTarget?.count) {
      goalX = strategicTarget.x - dir * 285;
      goalZ = strategicTarget.z;
      goalSpeedScale = 0.9;
    } else if (plan.mission === 'flank' && strategicTarget?.count) {
      const outerZ = plan.flankSign * 385;
      if (Math.abs(ownRegiment.z - outerZ) > 55) {
        goalX = -dir * 60;
        goalZ = outerZ;
      } else {
        goalX = strategicTarget.x + dir * 70;
        goalZ = strategicTarget.z + plan.flankSign * 75;
      }
    } else if (strategicTarget?.count) {
      const strategicDx = strategicTarget.x - ownRegiment.x;
      const strategicDz = strategicTarget.z - ownRegiment.z;
      const strategicDistance = Math.hypot(strategicDx, strategicDz) || 1;
      targetVx = strategicDx / strategicDistance * marchSpeed;
      targetVx = Math.abs(targetVx) < marchSpeed * 0.55 ? dir * marchSpeed * 0.55 : targetVx;
      const convergence = plan.mission === 'assault' || plan.mission === 'counterattack' ? 1 : 0.65;
      targetVz = strategicDz / strategicDistance * marchSpeed * convergence;
    }
    if (goalX !== null) {
      const goalDx = goalX - ownRegiment.x;
      const goalDz = goalZ - ownRegiment.z;
      const goalDistance = Math.hypot(goalDx, goalDz);
      if (goalDistance >= 8) {
        const commandedSpeed = Math.min(marchSpeed * goalSpeedScale, goalDistance * 0.7);
        targetVx = goalDx / goalDistance * commandedSpeed;
        targetVz = goalDz / goalDistance * commandedSpeed;
      }
    }
    if (plan.mission !== 'crowd') {
      const cohesion = nearestD2 < 2500 ? 0 : 0.32;
      targetVx += (desiredFormationX - u.x) * cohesion;
      targetVz += (desiredFormationZ - u.z) * cohesion;
    }

    if (u.type === 1) {
      const candidates = teamIndices[u.team === 0 ? 1 : 0];
      let rangedIndex = -1;
      let rangedD2 = 130000;
      const samples = Math.min(14, candidates.length);
      for (let s = 0; s < samples; s++) {
        const candidate = candidates[(rand() * candidates.length) | 0];
        const e = units[candidate];
        const dx = e.x - u.x;
        const dz = e.z - u.z;
        const d2 = dx * dx + dz * dz;
        if (d2 < rangedD2) {
          rangedD2 = d2;
          rangedIndex = candidate;
        }
      }
      if (rangedIndex >= 0) {
        const target = units[rangedIndex];
        const distance = Math.sqrt(rangedD2);
        if (distance < 340) {
          targetVx = distance < 90 ? -dir * 36 : targetVx * 0.16;
          targetVz += (desiredFormationZ - u.z) * 0.08;
          const commander = armyCommanders[u.team];
          const volleyActive = simTime <= commander.volleyUntil;
          const freeFire = commander.mode === 'crowd' && u.cooldown <= 0;
          const commandedVolley = commander.mode !== 'crowd' && volleyActive && u.lastVolley !== commander.volleyId;
          if ((freeFire || commandedVolley) && distance > 25) {
            fireArrow(u, rangedIndex, target);
            u.lastVolley = commander.volleyId;
            u.cooldown = freeFire ? SHOT_CADENCE : 0.8;
          }
        }
      }
    }

    if (nearestEnemy >= 0) {
      const enemy = units[nearestEnemy];
      const dx = enemy.x - u.x;
      const dz = enemy.z - u.z;
      const distance = Math.sqrt(nearestD2) || 1;
      if (distance < 11) {
        targetVx = -dx / distance * 3;
        targetVz = -dz / distance * 3;
        if (u.cooldown <= 0) {
          enemy.hp -= u.type ? 3 + rand() * 5 : 7 + rand() * 8;
          u.cooldown = 0.58 + rand() * 0.55;
          if (enemy.hp <= 0) killUnit(enemy);
        }
      } else if (distance < 45) {
        targetVx = dx / distance * 39;
        targetVz = dz / distance * 39;
      }
    }

    const localPressure = pressure - friends * 0.2;
    if (localPressure > 2) u.morale -= dt * 0.018 * localPressure;
    else u.morale = Math.min(1, u.morale + dt * 0.004);
    if (u.morale < 0.18) {
      targetVx = -dir * 50;
      targetVz += (rand() - 0.5) * 22;
    }

    targetVx += sepX * 23;
    targetVz += sepZ * 23;
    const response = Math.min(1, dt * 5.5);
    u.vx += (targetVx - u.vx) * response;
    u.vz += (targetVz - u.vz) * response;
    const speed = Math.hypot(u.vx, u.vz);
    if (speed > 54) {
      u.vx *= 54 / speed;
      u.vz *= 54 / speed;
    }
    u.x = Math.max(-WORLD_W / 2 + 8, Math.min(WORLD_W / 2 - 8, u.x + u.vx * dt));
    u.z = Math.max(-WORLD_D / 2 + 8, Math.min(WORLD_D / 2 - 8, u.z + u.vz * dt));
  }

  for (let i = arrows.length - 1; i >= 0; i--) {
    const a = arrows[i];
    a.life -= dt;
    a.x += a.vx * dt;
    a.y += a.vy * dt;
    a.z += a.vz * dt;
    a.vy -= 170 * dt;
    if (a.life <= 0 || a.y < 0) {
      const target = units[a.target];
      if (target?.alive && rand() < 0.68) {
        target.hp -= a.damage;
        if (target.hp <= 0) killUnit(target);
      }
      arrows.splice(i, 1);
    }
  }

  if (aliveBlue === 0 || aliveRed === 0 || simTime > 240) ended = true;
}

function getMesh(u) {
  if (u.team === 0) return u.type ? unitMeshes.blueArchers : unitMeshes.blueInfantry;
  return u.type ? unitMeshes.redArchers : unitMeshes.redInfantry;
}

function updateMeshes() {
  const counts = new Map(Object.values(unitMeshes).map(mesh => [mesh, 0]));
  for (let i = 0; i < unitCount; i++) {
    const u = units[i];
    const mesh = getMesh(u);
    const index = counts.get(mesh);
    dummy.position.set(u.x, u.alive ? 0 : 1.2, u.z);
    const heading = Number.isFinite(u.facing)
      ? Math.PI / 2 - u.facing
      : Math.atan2(u.vx, u.vz);
    dummy.rotation.set(u.alive ? 0 : Math.PI * 0.47 * u.fall, heading, 0);
    const bob = u.alive ? Math.sin(simTime * 8 + u.phase) * Math.min(0.7, Math.hypot(u.vx, u.vz) * 0.02) : 0;
    dummy.position.y += bob;
    dummy.scale.setScalar(u.alive ? 1 : Math.max(0.65, 1 - u.fall * 0.25));
    dummy.updateMatrix();
    mesh.setMatrixAt(index, dummy.matrix);
    counts.set(mesh, index + 1);
  }

  for (const mesh of Object.values(unitMeshes)) {
    mesh.count = counts.get(mesh);
    mesh.instanceMatrix.needsUpdate = true;
  }

  const groups = Array.from({ length: 2 }, () =>
    Array.from({ length: REGIMENTS }, () => ({ count: 0, x: 0, z: 0, front: null }))
  );
  for (let i = 0; i < unitCount; i++) {
    const u = units[i];
    if (!u.alive) continue;
    const g = groups[u.team][u.regiment];
    g.count++;
    g.z += u.z;
    if (g.front === null || (u.team === 0 ? u.x > g.front : u.x < g.front)) g.front = u.x;
  }
  for (const item of flagPoles) {
    const g = groups[item.team][item.regiment];
    item.holder.visible = g.count > 0;
    if (g.count) {
      const dir = item.team === 0 ? 1 : -1;
      item.holder.position.set(g.front + dir * 12, 0, g.z / g.count);
      item.holder.rotation.y = item.team === 0 ? 0 : Math.PI;
    }
  }

  const arrowCount = Math.min(arrows.length, MAX_ARROWS);
  for (let i = 0; i < arrowCount; i++) {
    const a = arrows[i];
    const speed = Math.hypot(a.vx, a.vy, a.vz) || 1;
    const offset = i * 6;
    arrowPositions[offset] = a.x;
    arrowPositions[offset + 1] = a.y;
    arrowPositions[offset + 2] = a.z;
    arrowPositions[offset + 3] = a.x - a.vx / speed * 12;
    arrowPositions[offset + 4] = a.y - a.vy / speed * 12;
    arrowPositions[offset + 5] = a.z - a.vz / speed * 12;
  }
  arrowGeometry.setDrawRange(0, arrowCount * 2);
  arrowGeometry.attributes.position.needsUpdate = true;
}

function updateHud() {
  let blue = 0;
  let red = 0;
  let contact = false;
  for (let i = 0; i < unitCount; i++) {
    const u = units[i];
    if (!u.alive) continue;
    if (u.team === 0) blue++; else red++;
    if (Math.abs(u.x) < 85) contact = true;
  }
  blueCountEl.textContent = blue.toLocaleString('ru-RU');
  redCountEl.textContent = red.toLocaleString('ru-RU');
  blueOrderEl.textContent = armyCommanders[0]?.label ?? 'ОБЩЕЕ НАСТУПЛЕНИЕ';
  redOrderEl.textContent = armyCommanders[1]?.label ?? 'ОБЩЕЕ НАСТУПЛЕНИЕ';
  const initial = Number(unitsInput.value);
  const volleyNow = armyCommanders.some(commander => commander.mode !== 'crowd' && simTime <= commander.volleyUntil);
  let phase = volleyNow && arrows.length ? 'ЛУЧНЫЙ ЗАЛП' : contact ? 'СХВАТКА' : arrows.length ? 'ПЕРЕСТРЕЛКА' : 'СБЛИЖЕНИЕ';
  if (ended) phase = blue === red ? 'НИЧЬЯ' : blue > red ? 'ПОБЕДА СИНИХ' : 'ПОБЕДА КРАСНЫХ';
  else if (blue < initial * 0.35 || red < initial * 0.35) phase = 'РАЗГРОМ';
  phaseEl.textContent = phase;
  const minutes = Math.floor(simTime / 60);
  const seconds = Math.floor(simTime % 60);
  battleTimeEl.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  fpsEl.textContent = `${shownFps} FPS`;
}

function resetCamera() {
  camera.position.set(0, 650, 850);
  controls.target.set(0, 0, 0);
  controls.update();
}

function updateFreeCamera(dt) {
  const speed = 320 * dt * Math.max(0.7, camera.position.distanceTo(controls.target) / 550);
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const move = new THREE.Vector3();
  if (keys.has('KeyW')) move.add(forward);
  if (keys.has('KeyS')) move.sub(forward);
  if (keys.has('KeyD')) move.add(right);
  if (keys.has('KeyA')) move.sub(right);
  if (keys.has('KeyE')) move.y += 1;
  if (keys.has('KeyQ')) move.y -= 1;
  if (move.lengthSq() > 0) {
    move.normalize().multiplyScalar(speed);
    camera.position.add(move);
    controls.target.add(move);
    camera.position.y = Math.max(16, camera.position.y);
    controls.target.x = THREE.MathUtils.clamp(controls.target.x, -WORLD_W, WORLD_W);
    controls.target.z = THREE.MathUtils.clamp(controls.target.z, -WORLD_D, WORLD_D);
  }
}

function changeStrategy(team, mode) {
  if (coreBattle) {
    coreBattle.setStrategy(team, mode);
    armyCommanders = coreBattle.commanders;
    regimentPlans = coreBattle.plans;
    return;
  }
  const commander = armyCommanders[team];
  if (!commander) return;
  commander.mode = mode;
  commander.nextDecision = simTime;
  commander.nextVolley = simTime + SHOT_CADENCE;
  commander.volleyUntil = -1;
  for (const plan of regimentPlans[team]) {
    plan.mission = mode === 'crowd' ? 'crowd' : 'advance';
    plan.order = plan.mission;
  }
}

function animate(now) {
  const realDt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  fpsFrames++;
  fpsTimer += realDt;
  if (fpsTimer >= 0.5) {
    shownFps = Math.round(fpsFrames / fpsTimer);
    fpsFrames = 0;
    fpsTimer = 0;
  }

  updateFreeCamera(realDt);
  controls.update();

  if (!paused) {
    const simulationSpeed = Number(speedInput.value);
    accumulator += realDt * simulationSpeed;
    const fixed = 1 / 30;
    const maxSteps = Math.max(8, Math.ceil(simulationSpeed * 1.5));
    let loops = 0;
    while (accumulator >= fixed && loops < maxSteps) {
      simulate(fixed);
      accumulator -= fixed;
      loops++;
    }
  }

  updateMeshes();
  updateHud();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

unitsInput.addEventListener('input', () => {
  unitsValue.textContent = Number(unitsInput.value).toLocaleString('ru-RU');
});
unitsInput.addEventListener('change', resetBattle);
speedInput.addEventListener('input', () => {
  speedValue.textContent = `${Number(speedInput.value).toLocaleString('ru-RU')}×`;
});
blueStrategyInput.addEventListener('change', () => changeStrategy(0, blueStrategyInput.value));
redStrategyInput.addEventListener('change', () => changeStrategy(1, redStrategyInput.value));
pauseButton.addEventListener('click', () => {
  paused = !paused;
  pauseButton.textContent = paused ? 'Продолжить' : 'Пауза';
});
resetButton.addEventListener('click', resetBattle);
cameraResetButton.addEventListener('click', resetCamera);

window.addEventListener('keydown', event => {
  if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE'].includes(event.code)) {
    keys.add(event.code);
    event.preventDefault();
  }
});
window.addEventListener('keyup', event => keys.delete(event.code));
window.addEventListener('blur', () => keys.clear());
window.addEventListener('resize', () => {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
});

resetBattle();
resetCamera();
requestAnimationFrame(animate);
requestAnimationFrame(() => loading.classList.add('hidden'));
