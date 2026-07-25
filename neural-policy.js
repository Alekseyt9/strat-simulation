export const POLICY_INPUTS = 48;
export const POLICY_HIDDEN_1 = 32;
export const POLICY_HIDDEN_2 = 24;
export const POLICY_VERSION = 3;
export const SECTOR_COUNT = 7;
export const POLICY_STANCES = ['hold', 'advance', 'assault', 'reserve', 'flank'];
export const POLICY_OUTPUTS = SECTOR_COUNT + POLICY_STANCES.length;

export function parameterCount() {
  return POLICY_INPUTS * POLICY_HIDDEN_1
    + POLICY_HIDDEN_1
    + POLICY_HIDDEN_1 * POLICY_HIDDEN_2
    + POLICY_HIDDEN_2
    + POLICY_HIDDEN_2 * POLICY_OUTPUTS
    + POLICY_OUTPUTS;
}

export function createPolicy(fill = 0) {
  return {
    version: POLICY_VERSION,
    inputs: POLICY_INPUTS,
    hidden: [POLICY_HIDDEN_1, POLICY_HIDDEN_2],
    outputs: POLICY_OUTPUTS,
    weights: new Array(parameterCount()).fill(fill)
  };
}

export function isCompatiblePolicy(policy) {
  return policy?.version === POLICY_VERSION
    && policy?.weights?.length === parameterCount();
}

export function evaluateOrders(features, policy) {
  if (!isCompatiblePolicy(policy)) {
    throw new Error(`Несовместимая нейрополитика: нужна версия ${POLICY_VERSION}`);
  }
  const weights = policy.weights;
  let cursor = 0;
  const first = new Array(POLICY_HIDDEN_1);
  for (let h = 0; h < POLICY_HIDDEN_1; h++) {
    let sum = 0;
    for (let input = 0; input < POLICY_INPUTS; input++) sum += features[input] * weights[cursor++];
    first[h] = Math.tanh(sum);
  }
  for (let h = 0; h < POLICY_HIDDEN_1; h++) first[h] = Math.tanh(first[h] + weights[cursor++]);

  const second = new Array(POLICY_HIDDEN_2);
  for (let h = 0; h < POLICY_HIDDEN_2; h++) {
    let sum = 0;
    for (let input = 0; input < POLICY_HIDDEN_1; input++) sum += first[input] * weights[cursor++];
    second[h] = Math.tanh(sum);
  }
  for (let h = 0; h < POLICY_HIDDEN_2; h++) second[h] = Math.tanh(second[h] + weights[cursor++]);

  const outputs = new Array(POLICY_OUTPUTS);
  for (let output = 0; output < POLICY_OUTPUTS; output++) {
    let sum = 0;
    for (let h = 0; h < POLICY_HIDDEN_2; h++) sum += second[h] * weights[cursor++];
    outputs[output] = sum;
  }
  for (let output = 0; output < POLICY_OUTPUTS; output++) outputs[output] += weights[cursor++];
  return {
    sectors: outputs.slice(0, SECTOR_COUNT),
    stances: outputs.slice(SECTOR_COUNT)
  };
}
