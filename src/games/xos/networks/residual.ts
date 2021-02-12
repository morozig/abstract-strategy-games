import * as tf from '@tensorflow/tfjs';
import {
    residualNetwork2D,
    denseLayer,
    convLayer2D
} from '../../../lib/networks';

const numFilters = 128;
const numLayers = 5;
const batchSize = 64;
const epochs = 30;
const learningRate = 0.01;
const dropout = 0.3;

interface LayersModelOptions {
  height: number;
  width: number;
  depth: number;
}

const createModel = (options: LayersModelOptions) => {
  const {
    height,
    width,
    depth
  } = options;
  const input = tf.input({
    shape: [height, width, depth]
  });

  let network = residualNetwork2D(input, {
    numLayers,
    numFilters,
    kernelSize: 3
  });

  let policy = convLayer2D(network, {
    name: 'policyConv',
    numFilters: 2,
    kernelSize: 1,
    padding: 'same'
  })

  policy = tf.layers.flatten(
  ).apply(policy) as tf.SymbolicTensor;

  policy = denseLayer(policy, {
    name: 'policyDense',
    units: 2 * height * width,
    dropout
  });

  policy = denseLayer(policy, {
    name: 'policyDenseHead',
    units: height * width,
    dropout
  });

  policy = tf.layers.activation({
    activation: 'softmax',
    name: 'policy'
  }).apply(policy) as tf.SymbolicTensor;

  let reward = convLayer2D(network, {
    name: 'rewardConv',
    numFilters: 1,
    kernelSize: 1,
    padding: 'same'
  });

  reward = tf.layers.flatten(
  ).apply(reward) as tf.SymbolicTensor;

  reward = denseLayer(reward, {
    name: 'rewardDense',
    units: 20,
    dropout
  });

  reward = denseLayer(reward, {
    name: 'rewardDenseHead',
    units: 1,
    dropout
  });

  reward = tf.layers.activation({
    activation: 'tanh',
    name: 'reward'
  }).apply(reward) as tf.SymbolicTensor;

  const model = tf.model(
    {
      inputs: input,
      outputs: [
        policy,
        reward
      ]
    }
  );
  return model;
};

export {
  createModel,
  batchSize,
  epochs,
  learningRate
};
