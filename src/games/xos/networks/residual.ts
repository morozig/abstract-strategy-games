import * as tf from '@tensorflow/tfjs';
import {
    residualNetwork2D,
    denseLayer,
    convLayer2D
} from '../../../lib/networks';

const numFilters = 128;
const numLayers = 5;
const batchSize = 64;
const epochs = 10;
const learningRate = 0.001;
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
    numFilters: 1,
    kernelSize: 1,
    padding: 'same'
  });

  policy = tf.layers.flatten(
  ).apply(policy) as tf.SymbolicTensor;

  policy = tf.layers.softmax(
  ).apply(policy) as tf.SymbolicTensor;


  let reward = convLayer2D(network, {
    name: 'rewardConv',
    numFilters,
    kernelSize: [
      height,
      width
    ],
    padding: 'valid'
  });

  reward = tf.layers.flatten(
  ).apply(reward) as tf.SymbolicTensor;

  reward = denseLayer(reward, {
    name: 'rewardDense',
    units: 1,
    dropout
  });

  reward = tf.layers.activation({
    activation: 'tanh'
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
