import { POLICY_INPUTS, POLICY_STANCES, SECTOR_COUNT } from './neural-policy.js';

export const V3_POLICY_VERSION = 1;
export const V3_EMBED = 64;
export const V3_ATTENTION = 32;
export const V3_MEMORY = 64;
export const V3_ACTOR = 64;

const SHAPES = {
  encoderWeight: [V3_EMBED, POLICY_INPUTS],
  encoderBias: [V3_EMBED],
  queryWeight: [V3_ATTENTION, V3_EMBED],
  queryBias: [V3_ATTENTION],
  keyWeight: [V3_ATTENTION, V3_EMBED],
  keyBias: [V3_ATTENTION],
  valueAttentionWeight: [V3_ATTENTION, V3_EMBED],
  valueAttentionBias: [V3_ATTENTION],
  mixWeight: [V3_EMBED, V3_EMBED + V3_ATTENTION],
  mixBias: [V3_EMBED],
  gruWeightInput: [V3_MEMORY * 3, V3_EMBED],
  gruWeightHidden: [V3_MEMORY * 3, V3_MEMORY],
  gruBiasInput: [V3_MEMORY * 3],
  gruBiasHidden: [V3_MEMORY * 3],
  actorWeight: [V3_ACTOR, V3_EMBED + V3_MEMORY],
  actorBias: [V3_ACTOR],
  sectorWeight: [SECTOR_COUNT, V3_ACTOR],
  sectorBias: [SECTOR_COUNT],
  stanceWeight: [POLICY_STANCES.length, V3_ACTOR],
  stanceBias: [POLICY_STANCES.length],
  criticWeight: [1, V3_MEMORY],
  criticBias: [1]
};

export function v3ParameterCount() {
  return Object.values(SHAPES).reduce(
    (total, shape) => total + shape.reduce((size, value) => size * value, 1),
    0
  );
}

export function createV3Policy() {
  const weights = {};
  for (const [name, shape] of Object.entries(SHAPES)) {
    weights[name] = new Array(shape.reduce((size, value) => size * value, 1)).fill(0);
  }
  return {
    version: V3_POLICY_VERSION,
    inputs: POLICY_INPUTS,
    regiments: 'variable',
    embed: V3_EMBED,
    attention: V3_ATTENTION,
    memory: V3_MEMORY,
    actor: V3_ACTOR,
    weights
  };
}

export function isCompatibleV3Policy(policy) {
  if (policy?.version !== V3_POLICY_VERSION || !policy.weights) return false;
  return Object.entries(SHAPES).every(([name, shape]) =>
    policy.weights[name]?.length === shape.reduce((size, value) => size * value, 1)
  );
}

function linear(input, weight, bias, outputs) {
  const result = new Array(outputs);
  for (let output = 0; output < outputs; output++) {
    let sum = bias[output];
    const offset = output * input.length;
    for (let index = 0; index < input.length; index++) sum += input[index] * weight[offset + index];
    result[output] = sum;
  }
  return result;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));
}

function gruCell(input, hidden, weights) {
  const inputGates = linear(input, weights.gruWeightInput, weights.gruBiasInput, V3_MEMORY * 3);
  const hiddenGates = linear(hidden, weights.gruWeightHidden, weights.gruBiasHidden, V3_MEMORY * 3);
  const next = new Array(V3_MEMORY);
  for (let index = 0; index < V3_MEMORY; index++) {
    const reset = sigmoid(inputGates[index] + hiddenGates[index]);
    const update = sigmoid(inputGates[V3_MEMORY + index] + hiddenGates[V3_MEMORY + index]);
    const candidate = Math.tanh(
      inputGates[V3_MEMORY * 2 + index] + reset * hiddenGates[V3_MEMORY * 2 + index]
    );
    next[index] = (1 - update) * candidate + update * hidden[index];
  }
  return next;
}

export function evaluateV3Orders(regimentFeatures, activeMask, previousHidden, policy) {
  if (!isCompatibleV3Policy(policy)) {
    throw new Error(`Несовместимая политика Commander V3: нужна версия ${V3_POLICY_VERSION}`);
  }
  const w = policy.weights;
  const regimentCount = regimentFeatures.length;
  const encoded = regimentFeatures.map(features =>
    linear(features, w.encoderWeight, w.encoderBias, V3_EMBED).map(Math.tanh)
  );
  const queries = encoded.map(token => linear(token, w.queryWeight, w.queryBias, V3_ATTENTION));
  const keys = encoded.map(token => linear(token, w.keyWeight, w.keyBias, V3_ATTENTION));
  const values = encoded.map(token =>
    linear(token, w.valueAttentionWeight, w.valueAttentionBias, V3_ATTENTION)
  );
  const mixed = encoded.map((token, regiment) => {
    const scores = keys.map((key, target) => {
      if (!activeMask[target]) return -1e9;
      let score = 0;
      for (let index = 0; index < V3_ATTENTION; index++) {
        score += queries[regiment][index] * key[index];
      }
      return score / Math.sqrt(V3_ATTENTION);
    });
    const maximum = Math.max(...scores);
    const probabilities = scores.map(score => score < -1e8 ? 0 : Math.exp(score - maximum));
    const total = probabilities.reduce((sum, value) => sum + value, 0) || 1;
    const context = new Array(V3_ATTENTION).fill(0);
    for (let target = 0; target < regimentCount; target++) {
      const probability = probabilities[target] / total;
      for (let index = 0; index < V3_ATTENTION; index++) {
        context[index] += probability * values[target][index];
      }
    }
    return linear([...token, ...context], w.mixWeight, w.mixBias, V3_EMBED).map(Math.tanh);
  });
  const pooled = new Array(V3_EMBED).fill(0);
  const active = Math.max(1, activeMask.filter(Boolean).length);
  for (let regiment = 0; regiment < regimentCount; regiment++) {
    if (!activeMask[regiment]) continue;
    for (let index = 0; index < V3_EMBED; index++) pooled[index] += mixed[regiment][index] / active;
  }
  const hidden = previousHidden?.length === V3_MEMORY
    ? previousHidden
    : new Array(V3_MEMORY).fill(0);
  const nextHidden = gruCell(pooled, hidden, w);
  const outputs = mixed.map(token => {
    const actor = linear([...token, ...nextHidden], w.actorWeight, w.actorBias, V3_ACTOR).map(Math.tanh);
    return {
      sectors: linear(actor, w.sectorWeight, w.sectorBias, SECTOR_COUNT),
      stances: linear(actor, w.stanceWeight, w.stanceBias, POLICY_STANCES.length)
    };
  });
  return {
    outputs,
    value: linear(nextHidden, w.criticWeight, w.criticBias, 1)[0],
    hidden: nextHidden
  };
}

export function mixedDistribution(logits, allowed, exploration = 0) {
  let maximum = -Infinity;
  for (const index of allowed) maximum = Math.max(maximum, logits[index]);
  const probabilities = new Array(logits.length).fill(0);
  let total = 0;
  for (const index of allowed) {
    probabilities[index] = Math.exp(Math.max(-30, logits[index] - maximum));
    total += probabilities[index];
  }
  const uniform = 1 / Math.max(1, allowed.length);
  for (const index of allowed) {
    probabilities[index] = (1 - exploration) * probabilities[index] / (total || 1)
      + exploration * uniform;
  }
  return probabilities;
}

export function chooseV3Action(probabilities, random = null) {
  if (!random) {
    let best = 0;
    for (let index = 1; index < probabilities.length; index++) {
      if (probabilities[index] > probabilities[best]) best = index;
    }
    return best;
  }
  let threshold = random();
  for (let index = 0; index < probabilities.length; index++) {
    threshold -= probabilities[index];
    if (threshold <= 0 && probabilities[index] > 0) return index;
  }
  return probabilities.findLastIndex(value => value > 0);
}
