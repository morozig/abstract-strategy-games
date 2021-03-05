import * as tf from '@tensorflow/tfjs';

const numFilters = 128;
const epochs = 10;
const dropout = 0.3;
const learningRate = 0.05;
const batchSize = 1024;

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

  let network = tf.layers.conv2d({
    kernelSize: 3,
    filters: numFilters,
    strides: 1,
    padding: 'same',
    useBias: false,
      // kernelRegularizer: 'l1l2'
  }).apply(input) as tf.SymbolicTensor;

  network = tf.layers.batchNormalization({
    axis: 3
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.activation({
    activation: 'relu'
  }).apply(network) as tf.SymbolicTensor;

  network = tf.layers.dropout({
    rate: dropout
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.conv2d({
    kernelSize: 3,
    filters: numFilters,
    strides: 1,
    padding: 'same',
    useBias: false,
      // kernelRegularizer: 'l1l2'
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.batchNormalization({
    axis: 3
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.activation({
    activation: 'relu'
  }).apply(network) as tf.SymbolicTensor;

  network = tf.layers.dropout({
    rate: dropout
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.conv2d({
    kernelSize: 3,
    filters: numFilters,
    strides: 1,
    padding: 'same',
    useBias: false,
    // kernelRegularizer: 'l1l2'
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.batchNormalization({
    axis: 3
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.activation({
    activation: 'relu'
  }).apply(network) as tf.SymbolicTensor;

  network = tf.layers.dropout({
    rate: dropout
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.conv2d({
    kernelSize: 3,
    filters: numFilters,
    strides: 1,
    padding: 'same',
    useBias: false,
      // kernelRegularizer: 'l1l2'
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.batchNormalization({
    axis: 3
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.activation({
    activation: 'relu'
  }).apply(network) as tf.SymbolicTensor;

  network = tf.layers.dropout({
    rate: dropout
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.conv2d({
    kernelSize: 1,
    filters: 1,
    strides: 1,
    padding: 'same',
    useBias: false,
      // kernelRegularizer: 'l1l2'
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.batchNormalization({
    axis: 3
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.activation({
    activation: 'relu'
  }).apply(network) as tf.SymbolicTensor;

  network = tf.layers.flatten(
  ).apply(network) as tf.SymbolicTensor;

  // network = tf.layers.dense({
  //   units: 2 * numFilters,
  //   useBias: false,
  //     // kernelRegularizer: 'l1l2'
  // }).apply(network) as tf.SymbolicTensor;
  // network = tf.layers.batchNormalization({
  //   axis: 1
  // }).apply(network) as tf.SymbolicTensor;
  // network = tf.layers.activation({
  //   activation: 'relu'
  // }).apply(network) as tf.SymbolicTensor;

  // network = tf.layers.dropout({
  //   rate: dropout
  // }).apply(network) as tf.SymbolicTensor;

  network = tf.layers.dropout({
    rate: dropout
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.dense({
    units: numFilters,
    useBias: false,
      // kernelRegularizer: 'l1l2'
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.batchNormalization({
    axis: 1
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.activation({
    activation: 'relu'
  }).apply(network) as tf.SymbolicTensor;
  // network = tf.layers.dropout({
  //     rate: dropout
  // }).apply(network) as tf.SymbolicTensor;


  let policy = tf.layers.dense({
    units: height * width
  }).apply(network) as tf.SymbolicTensor;
  policy = tf.layers.activation({
    activation: 'softmax',
    name: 'policy'
  }).apply(policy) as tf.SymbolicTensor;

  let reward = tf.layers.dense({
    units: 1
  }).apply(network) as tf.SymbolicTensor;
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
