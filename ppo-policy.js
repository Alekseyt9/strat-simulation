import { POLICY_INPUTS, POLICY_STANCES, SECTOR_COUNT } from './neural-policy.js';

export const PPO_POLICY_VERSION = 1;
export const PPO_ENCODER = 64;
export const PPO_MEMORY = 48;

const SHAPES = {
  encoderWeight: [PPO_ENCODER, POLICY_INPUTS],
  encoderBias: [PPO_ENCODER],
  gruWeightInput: [PPO_MEMORY * 3, PPO_ENCODER],
  gruWeightHidden: [PPO_MEMORY * 3, PPO_MEMORY],
  gruBiasInput: [PPO_MEMORY * 3],
  gruBiasHidden: [PPO_MEMORY * 3],
  sectorWeight: [SECTOR_COUNT, PPO_MEMORY],
  sectorBias: [SECTOR_COUNT],
  stanceWeight: [POLICY_STANCES.length, PPO_MEMORY],
  stanceBias: [POLICY_STANCES.length],
  valueWeight: [1, PPO_MEMORY],
  valueBias: [1]
};

export function ppoParameterCount() {
  return Object.values(SHAPES).reduce(
    (total, shape) => total + shape.reduce((size, value) => size * value, 1),
    0
  );
}

export function createPpoPolicy() {
  const weights = {};
  for (const [name, shape] of Object.entries(SHAPES)) {
    weights[name] = new Array(shape.reduce((size, value) => size * value, 1)).fill(0);
  }
  return {
    version: PPO_POLICY_VERSION,
    inputs: POLICY_INPUTS,
    encoder: PPO_ENCODER,
    memory: PPO_MEMORY,
    sectors: SECTOR_COUNT,
    stances: POLICY_STANCES.length,
    weights
  };
}

export function isCompatiblePpoPolicy(policy) {
  if (policy?.version !== PPO_POLICY_VERSION || !policy.weights) return false;
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

export function evaluatePpoOrders(features, previousHidden, policy) {
  if (!isCompatiblePpoPolicy(policy)) {
    throw new Error(`Несовместимая PPO-политика: нужна версия ${PPO_POLICY_VERSION}`);
  }
  const w = policy.weights;
  const encoded = linear(features, w.encoderWeight, w.encoderBias, PPO_ENCODER).map(Math.tanh);
  const hidden = previousHidden?.length === PPO_MEMORY
    ? previousHidden
    : new Array(PPO_MEMORY).fill(0);
  const inputGates = linear(encoded, w.gruWeightInput, w.gruBiasInput, PPO_MEMORY * 3);
  const hiddenGates = linear(hidden, w.gruWeightHidden, w.gruBiasHidden, PPO_MEMORY * 3);
  const nextHidden = new Array(PPO_MEMORY);
  for (let index = 0; index < PPO_MEMORY; index++) {
    const reset = sigmoid(inputGates[index] + hiddenGates[index]);
    const update = sigmoid(inputGates[PPO_MEMORY + index] + hiddenGates[PPO_MEMORY + index]);
    const candidate = Math.tanh(
      inputGates[PPO_MEMORY * 2 + index] + reset * hiddenGates[PPO_MEMORY * 2 + index]
    );
    nextHidden[index] = (1 - update) * candidate + update * hidden[index];
  }
  return {
    sectors: linear(nextHidden, w.sectorWeight, w.sectorBias, SECTOR_COUNT),
    stances: linear(nextHidden, w.stanceWeight, w.stanceBias, POLICY_STANCES.length),
    value: linear(nextHidden, w.valueWeight, w.valueBias, 1)[0],
    hidden: nextHidden
  };
}

export function maskedDistribution(logits, allowed) {
  let maximum = -Infinity;
  for (const index of allowed) maximum = Math.max(maximum, logits[index]);
  const probabilities = new Array(logits.length).fill(0);
  let total = 0;
  for (const index of allowed) {
    const probability = Math.exp(Math.max(-30, logits[index] - maximum));
    probabilities[index] = probability;
    total += probability;
  }
  for (const index of allowed) probabilities[index] /= total || 1;
  return probabilities;
}

export function chooseCategorical(probabilities, random = null) {
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
