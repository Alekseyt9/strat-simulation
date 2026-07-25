export const V4_POLICY_VERSION = 2;
export const V4_INPUTS = 48;
export const V4_EMBED = 64;
export const V4_ATTENTION = 32;
export const V4_MEMORY = 64;
export const V4_DOCTRINES = [
  'compact_hold',
  'mass_advance',
  'mass_assault',
  'elastic',
  'left_hook',
  'right_hook',
  'counterattack',
  'encircle'
];
export const V4_FOCUS_SECTORS = 7;
export const V4_FIRE_MODES = ['independent', 'volley', 'hold_fire'];

const SHAPES = {
  encoderWeight: [V4_EMBED, V4_INPUTS],
  encoderBias: [V4_EMBED],
  queryWeight: [V4_ATTENTION, V4_EMBED],
  queryBias: [V4_ATTENTION],
  keyWeight: [V4_ATTENTION, V4_EMBED],
  keyBias: [V4_ATTENTION],
  valueAttentionWeight: [V4_ATTENTION, V4_EMBED],
  valueAttentionBias: [V4_ATTENTION],
  mixWeight: [V4_EMBED, V4_EMBED + V4_ATTENTION],
  mixBias: [V4_EMBED],
  gruWeightInput: [V4_MEMORY * 3, V4_EMBED],
  gruWeightHidden: [V4_MEMORY * 3, V4_MEMORY],
  gruBiasInput: [V4_MEMORY * 3],
  gruBiasHidden: [V4_MEMORY * 3],
  doctrineWeight: [V4_DOCTRINES.length, V4_MEMORY],
  doctrineBias: [V4_DOCTRINES.length],
  focusWeight: [V4_FOCUS_SECTORS, V4_MEMORY],
  focusBias: [V4_FOCUS_SECTORS],
  fireWeight: [V4_FIRE_MODES.length, V4_MEMORY],
  fireBias: [V4_FIRE_MODES.length],
  criticWeight: [1, V4_MEMORY],
  criticBias: [1]
};

export function v4ParameterCount() {
  return Object.values(SHAPES).reduce(
    (total, shape) => total + shape.reduce((size, value) => size * value, 1),
    0
  );
}

export function createV4Policy() {
  const weights = {};
  for (const [name, shape] of Object.entries(SHAPES)) {
    weights[name] = new Array(shape.reduce((size, value) => size * value, 1)).fill(0);
  }
  // Safe doctrine prior: compact central defense. It changes orders, not physics.
  weights.doctrineBias[0] = 1.5;
  weights.focusBias[3] = 1.5;
  weights.fireBias[0] = 1;
  return {
    version: V4_POLICY_VERSION,
    inputs: V4_INPUTS,
    regiments: 'variable',
    embed: V4_EMBED,
    attention: V4_ATTENTION,
    memory: V4_MEMORY,
    doctrines: V4_DOCTRINES,
    focusSectors: V4_FOCUS_SECTORS,
    fireModes: V4_FIRE_MODES,
    weights
  };
}

export function isCompatibleV4Policy(policy) {
  if (policy?.version !== V4_POLICY_VERSION || !policy.weights) return false;
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
  const inputGates = linear(input, weights.gruWeightInput, weights.gruBiasInput, V4_MEMORY * 3);
  const hiddenGates = linear(hidden, weights.gruWeightHidden, weights.gruBiasHidden, V4_MEMORY * 3);
  const next = new Array(V4_MEMORY);
  for (let index = 0; index < V4_MEMORY; index++) {
    const reset = sigmoid(inputGates[index] + hiddenGates[index]);
    const update = sigmoid(inputGates[V4_MEMORY + index] + hiddenGates[V4_MEMORY + index]);
    const candidate = Math.tanh(
      inputGates[V4_MEMORY * 2 + index] + reset * hiddenGates[V4_MEMORY * 2 + index]
    );
    next[index] = (1 - update) * candidate + update * hidden[index];
  }
  return next;
}

export function evaluateV4(regimentFeatures, activeMask, previousHidden, policy) {
  if (!isCompatibleV4Policy(policy)) {
    throw new Error(`Несовместимая политика Commander V4: нужна версия ${V4_POLICY_VERSION}`);
  }
  const w = policy.weights;
  const count = regimentFeatures.length;
  const encoded = regimentFeatures.map(features =>
    linear(features, w.encoderWeight, w.encoderBias, V4_EMBED).map(Math.tanh)
  );
  const queries = encoded.map(token => linear(token, w.queryWeight, w.queryBias, V4_ATTENTION));
  const keys = encoded.map(token => linear(token, w.keyWeight, w.keyBias, V4_ATTENTION));
  const values = encoded.map(token =>
    linear(token, w.valueAttentionWeight, w.valueAttentionBias, V4_ATTENTION)
  );
  const mixed = encoded.map((token, regiment) => {
    const scores = keys.map((key, target) => {
      if (!activeMask[target]) return -1e9;
      let score = 0;
      for (let index = 0; index < V4_ATTENTION; index++) {
        score += queries[regiment][index] * key[index];
      }
      return score / Math.sqrt(V4_ATTENTION);
    });
    const maximum = Math.max(...scores);
    const probabilities = scores.map(score => score < -1e8 ? 0 : Math.exp(score - maximum));
    const total = probabilities.reduce((sum, value) => sum + value, 0) || 1;
    const context = new Array(V4_ATTENTION).fill(0);
    for (let target = 0; target < count; target++) {
      const probability = probabilities[target] / total;
      for (let index = 0; index < V4_ATTENTION; index++) {
        context[index] += probability * values[target][index];
      }
    }
    return linear([...token, ...context], w.mixWeight, w.mixBias, V4_EMBED).map(Math.tanh);
  });
  const pooled = new Array(V4_EMBED).fill(0);
  const active = Math.max(1, activeMask.filter(Boolean).length);
  for (let regiment = 0; regiment < count; regiment++) {
    if (!activeMask[regiment]) continue;
    for (let index = 0; index < V4_EMBED; index++) pooled[index] += mixed[regiment][index] / active;
  }
  const hidden = previousHidden?.length === V4_MEMORY
    ? previousHidden
    : new Array(V4_MEMORY).fill(0);
  const nextHidden = gruCell(pooled, hidden, w);
  return {
    doctrines: linear(
      nextHidden, w.doctrineWeight, w.doctrineBias, V4_DOCTRINES.length
    ),
    focus: linear(nextHidden, w.focusWeight, w.focusBias, V4_FOCUS_SECTORS),
    fire: linear(nextHidden, w.fireWeight, w.fireBias, V4_FIRE_MODES.length),
    value: linear(nextHidden, w.criticWeight, w.criticBias, 1)[0],
    hidden: nextHidden
  };
}

export function v4Distribution(logits, exploration = 0) {
  const maximum = Math.max(...logits);
  const raw = logits.map(value => Math.exp(Math.max(-30, value - maximum)));
  const total = raw.reduce((sum, value) => sum + value, 0) || 1;
  const uniform = 1 / logits.length;
  return raw.map(value => (1 - exploration) * value / total + exploration * uniform);
}

export function chooseV4(probabilities, random = null) {
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
    if (threshold <= 0) return index;
  }
  return probabilities.length - 1;
}
