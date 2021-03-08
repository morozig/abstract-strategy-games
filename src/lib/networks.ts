import * as tf from '@tensorflow/tfjs';

const convLayer2D = (
  input: tf.SymbolicTensor,
  options: {
    numFilters: number;
    kernelSize?: number | number[];
    name: string;
    noActivation?: boolean;
    padding?: 'same' | 'valid'
  }
) => {
  const kernelSize = options.kernelSize || 3;
  const padding = options.padding || 'same';
  const useActivation = !options.noActivation;
  let network = tf.layers.conv2d({
    kernelSize,
    filters: options.numFilters,
    strides: 1,
    padding,
    name: options.name,
    useBias: false
  }).apply(input) as tf.SymbolicTensor;
  network = tf.layers.batchNormalization({
    name: `${options.name}_bn`,
    axis: 3
  })
    .apply(network) as tf.SymbolicTensor;
  if (useActivation) {
    network = tf.layers.activation({
      activation: 'relu'
    }).apply(network) as tf.SymbolicTensor;
  }
  return network;
};

const residualLayer2D = (
  input: tf.SymbolicTensor,
  options: {
    id: number;
    numFilters: number;
    kernelSize?: number;
    namePrefix?: string;
  }
) => {
  const namePrefix = options.namePrefix ?
    `${options.namePrefix}_` : '';
  let network = convLayer2D(input, {
    name: `${namePrefix}residual${options.id}_conv2d1`,
    numFilters: options.numFilters,
    kernelSize: options.kernelSize
  });
  network = convLayer2D(network, {
    name: `${namePrefix}residual${options.id}_conv2d2`,
    numFilters: options.numFilters,
    kernelSize: options.kernelSize,
    noActivation: true
  });

  network = tf.layers.add()
    .apply([network, input]) as tf.SymbolicTensor;;
  network = tf.layers.activation({
    activation: 'relu'
  }).apply(network) as tf.SymbolicTensor;;
  return network;
};

const residualNetwork2D = (
  input: tf.SymbolicTensor,
  options: {
    numLayers: number;
    numFilters: number;
    kernelSize?: number;
    namePrefix?: string;
  }
) => {
  const namePrefix = options.namePrefix ?
    `${options.namePrefix}_` : '';
  let network = convLayer2D(input, {
    name: `${namePrefix}residual0_conv2d`,
    numFilters: options.numFilters,
    kernelSize: options.kernelSize
  });

  for (let i = 1; i <= options.numLayers; i++) {
    network = residualLayer2D(network, {
      id: i,
      numFilters: options.numFilters,
      kernelSize: options.kernelSize,
      namePrefix: options.namePrefix
    })
  }
  return network;
};

const denseLayer = (
  input: tf.SymbolicTensor,
  options: {
    units: number;
    dropout: number;
    name: string;
  }
) => {
  let network = tf.layers.dense({
    units: options.units,
    name: options.name,
    useBias: false
  }).apply(input) as tf.SymbolicTensor;
  network = tf.layers.batchNormalization({
    name: `${options.name}_bn`,
    axis: 1
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.activation({
    activation: 'relu'
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.dropout({
    rate: options.dropout
  }).apply(network) as tf.SymbolicTensor;
  return network;
};

const copyWeights = (from: tf.LayersModel, to: tf.LayersModel) => {
  for (let sourceLayer of from.layers) {
    const sourceWeights = sourceLayer.getWeights();
    if (!sourceWeights || sourceWeights.length <= 0) {
      continue;
    }
    if (!to.layers.find(
      targetLayer => sourceLayer.name === targetLayer.name
    )){
      continue;
    }
    try {
      const targetLayer = to.getLayer(sourceLayer.name);
      targetLayer.setWeights(sourceWeights);
    }
    catch (err) {
      console.log(`Failed to copy layer ${sourceLayer.name}: ${err}`);
    }
  }
};

const countResidualLayers = (model: tf.LayersModel) => {
  const totalResidualConvLayers = model
    .layers
    .filter(layer => layer.name.startsWith('residual'))
    .length;
  return (totalResidualConvLayers - 2) / 4;
};

const kld = (
  labels: tf.Tensor|tf.TensorLike,
  predictions: tf.Tensor|tf.TensorLike,
  weights?: tf.Tensor|tf.TensorLike,
  epsilon = 1e-7,
  reduction = tf.Reduction.SUM_BY_NONZERO_WEIGHTS
): tf.Tensor => {
  const yPred = tf.clipByValue(predictions, epsilon, 1);
  const yTrue = tf.clipByValue(labels, epsilon, 1);
  const loss = tf.sum(tf.mul(
    yTrue,
    tf.log( tf.div( yTrue, yPred ) )
  ), -1);
  return tf.losses.computeWeightedLoss(loss, weights, reduction);
}

export {
  convLayer2D,
  residualNetwork2D,
  copyWeights,
  countResidualLayers,
  denseLayer,
  kld
}