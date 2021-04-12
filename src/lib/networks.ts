import * as tf from '@tensorflow/tfjs';
import { Shape } from '@tensorflow/tfjs';

const convLayer2D = (
  input: tf.SymbolicTensor,
  options: {
    numFilters: number;
    kernelSize?: number | number[];
    name: string;
    noActivation?: boolean;
    padding?: 'same' | 'valid';
    dropout?: number;
  }
) => {
  const kernelSize = options.kernelSize || 3;
  const padding = options.padding || 'same';
  const useActivation = !options.noActivation;
  let network = input;
  if (options.dropout) {
    network = tf.layers.dropout({
      rate: options.dropout
    }).apply(network) as tf.SymbolicTensor;
  }
  network = tf.layers.conv2d({
    kernelSize,
    filters: options.numFilters,
    strides: 1,
    padding,
    name: options.name,
    useBias: false
  }).apply(network) as tf.SymbolicTensor;
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
    dropout?: number;
  }
) => {
  const namePrefix = options.namePrefix ?
    `${options.namePrefix}_` : '';
  let network = convLayer2D(input, {
    name: `${namePrefix}residual${options.id}_conv2d1`,
    numFilters: options.numFilters,
    kernelSize: options.kernelSize,
    dropout: options.dropout
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
    dropout?: number;
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
    noActivation?: boolean;
  }
) => {
  const useActivation = !options.noActivation;
  let network = tf.layers.dropout({
    rate: options.dropout
  }).apply(input) as tf.SymbolicTensor;
  network = tf.layers.dense({
    units: options.units,
    name: options.name,
    useBias: false
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.batchNormalization({
    name: `${options.name}_bn`,
    axis: 1
  }).apply(network) as tf.SymbolicTensor;
  if (useActivation) {
    network = tf.layers.activation({
      activation: 'relu'
    }).apply(network) as tf.SymbolicTensor;
  }
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

const scaledDotProductAttention = (
  q: tf.Tensor,
  k: tf.Tensor,
  v: tf.Tensor,
  mask?: tf.Tensor
) => {
  const matmulQueriesKeys = tf.matMul(q, k, false, true);
  const keysShape = k.shape;
  const keysDim = keysShape[keysShape.length - 1];
  let scaledAttentionLogits = tf.div(
    matmulQueriesKeys,
    Math.sqrt(keysDim)
  );

  if (mask) {
    scaledAttentionLogits = tf.add(
      scaledAttentionLogits,
      tf.mul(mask, -1e9)
    )
  }

  const attentionWeights = tf.softmax(scaledAttentionLogits, -1);
  const output = tf.matMul(attentionWeights, v);
  return [
    output,
    attentionWeights
  ];
};

interface MultiHeadAttentionLayerArgs {
  dim: number;
  numHeads: number;
  dropout?: number;
  namePrefix?: string;
};

class MultiHeadAttention extends tf.layers.Layer {
  private dim: number;
  private numHeads: number;
  private dropout: number;
  private namePrefix: string;
  private depth: number;
  private wq: tf.layers.Layer;
  private wk: tf.layers.Layer;
  private wv: tf.layers.Layer;
  private attentionDropout: tf.layers.Layer;
  private dense: tf.layers.Layer;

  constructor(args: MultiHeadAttentionLayerArgs) {
    super(args as any);
    this.dim = args.dim;
    this.numHeads = args.numHeads;
    this.dropout = args.dropout || 0;
    this.namePrefix = args.namePrefix || '';
    tf.util.assert(
      this.dim % this.numHeads === 0,
      () => 'dim = numHeads * n'
    );
    this.depth = Math.floor(this.dim / this.numHeads);

    this.wq  = tf.layers.dense({
      units: this.dim,
      name: `${this.namePrefix}wq`
    });
    this.wk  = tf.layers.dense({
      units: this.dim,
      name: `${this.namePrefix}wk`
    });
    this.wv  = tf.layers.dense({
      units: this.dim,
      name: `${this.namePrefix}wv`
    });
    this.attentionDropout = tf.layers.dropout({
      rate: this.dropout
    })
    this.dense  = tf.layers.dense({
      units: this.dim,
      name: `${this.namePrefix}all`
    });
  }
  private splitHeads(x: tf.Tensor, batchSize: number) {
    const split = tf.reshape(x, [
      batchSize, -1, this.numHeads, this.depth
    ]);
    const out = tf.transpose(split, [0, 2, 1, 3]);
    return out;
  }
  computeOutputShape(inputShapes: Shape[]) {
    const qShape = inputShapes[0];
    const vShape = inputShapes[2];

    const batchSize = qShape[0];
    const outputShape = [
      batchSize,
      qShape[1],
      vShape[2]
    ];

    const attentionShape = [
      batchSize,
      this.numHeads,
      qShape[1],
      vShape[2]
    ];

    const outputShapes = [
      outputShape,
      attentionShape
    ];
    return outputShapes;
  }
  call(inputs: tf.Tensor[]){
    return tf.tidy(() => {
      let [
        q,
        k,
        v,
        mask = undefined
      ] = inputs;
      const batchSize = q.shape[0];
      q = this.wq.apply(q) as tf.Tensor;
      k = this.wk.apply(k) as tf.Tensor;
      v = this.wv.apply(v) as tf.Tensor;
  
      q = this.splitHeads(q, batchSize);
      k = this.splitHeads(k, batchSize);
      v = this.splitHeads(v, batchSize);
  
      let [
        scaledAttention,
        attentionWeights
      ] = scaledDotProductAttention(q, k, v, mask);
  
      scaledAttention = this.attentionDropout.apply(
        scaledAttention
      ) as tf.Tensor;
  
      scaledAttention = tf.transpose(scaledAttention, [0, 2, 1, 3]);
      const concatAttention = tf.reshape(scaledAttention, [
        batchSize, -1, this.dim
      ]);
      const output = this.dense.apply(concatAttention) as tf.Tensor;
      return [
        output,
        attentionWeights
      ];
    });
  }
  static get className() {
    return 'MultiHeadAttention';
  }
};

const mha = (
  inputs: tf.SymbolicTensor[],
  options: MultiHeadAttentionLayerArgs
) => {
  return (
    new MultiHeadAttention(options)
  ).apply(inputs) as tf.SymbolicTensor[];
};

interface TileEncoderLayerArgs {
  numTiles: number;
  dim: number;
  namePrefix?: string;
};

class TileEncoder extends tf.layers.Layer {
  private numTiles: number;
  private dim: number;
  private namePrefix: string;

  private projection: tf.layers.Layer;
  private positionEmbedding : tf.layers.Layer;

  constructor(args: TileEncoderLayerArgs) {
    super(args as any);
    this.numTiles = args.numTiles;
    this.dim = args.dim;
    this.namePrefix = args.namePrefix || '';

    this.projection  = tf.layers.dense({
      units: this.dim,
      name: `${this.namePrefix}proj`
    });
    this.positionEmbedding  = tf.layers.embedding({
      inputDim: this.numTiles,
      outputDim: this.dim,
      name: `${this.namePrefix}position`
    });
  }
  computeOutputShape(inputShape: Shape) {
    const batchSize = inputShape[0];
    const outputShape = [
      batchSize,
      this.numTiles,
      this.dim
    ];
    return outputShape;
  }
  call(inputs: tf.Tensor | tf.Tensor[]){
    const input = Array.isArray(inputs) ?
      inputs[0] : inputs;
    return tf.tidy(() => {
      const batchSize = input.shape[0];
      const sequence = tf.reshape(input, [
        batchSize, this.numTiles, -1
      ]);
  
      const projections = this.projection.apply(sequence) as tf.Tensor;
  
      const positions = tf.range(0, this.numTiles, 1);
      const positionEmbeddings = this.positionEmbedding.apply(positions) as tf.Tensor;
  
      const output = tf.add(projections, positionEmbeddings);
      return output;
    });
  }
  static get className() {
    return 'TileEncoder';
  }
};

const tileEncoder = (
  input: tf.SymbolicTensor,
  options: TileEncoderLayerArgs
) => {
  return (
    new TileEncoder(options)
  ).apply(input) as tf.SymbolicTensor;
};

const mlp = (
  input: tf.SymbolicTensor,
  options: {
    layerUnits: number[];
    dropout?: number;
    namePrefix?: string;
  }
) => {
  const namePrefix = options.namePrefix || '';
  let network = input;
  for (let layerId in options.layerUnits) {
    const units = options.layerUnits[layerId];
    network = tf.layers.dense({
      units,
      name: `${namePrefix}${layerId}`,
      activation: 'relu'
    }).apply(network) as tf.SymbolicTensor;
    network = tf.layers.dropout({
      rate: options.dropout || 0
    }).apply(network) as tf.SymbolicTensor;
  }
  return network;
};

const transformer = (
  input: tf.SymbolicTensor,
  options: {
    numTiles: number;
    dim: number;
    numHeads: number;
    mlpUnits: number[];
    numLayers: number;
    dropout?: number;
    headMlpUnits: number[];
    headDropout?: number;
    headDim: number;
    namePrefix?: string;
  }
) => {
  const {
    numTiles,
    dim,
    numHeads,
    mlpUnits: layerUnits,
    numLayers,
    dropout = 0,
    headMlpUnits,
    headDropout = 0,
    headDim,
    namePrefix = ''
  } = options;

  // console.log('input', input.shape);
  let network = tileEncoder(input, {
    dim,
    numTiles,
    namePrefix: `${namePrefix}te`
  });
  // console.log('tileEncoder', network.shape);
  // return network;

  for (let i = 1; i <= numLayers; i++) {
    const layerId = i <= numLayers / 2 ?
      `${i}` :
      `${i - numLayers - 1}`;
    const x1 = tf.layers.layerNormalization({
      name: `${namePrefix}${layerId}ln1`
    }).apply(network) as tf.SymbolicTensor;
    const [ attentionOutput ] = mha([ x1, x1, x1], {
      dim,
      numHeads,
      dropout,
      namePrefix: `${namePrefix}${layerId}mha`
    });
    const x2 = tf.layers.add(
    ).apply([ attentionOutput, network]) as tf.SymbolicTensor;
    let x3 = tf.layers.layerNormalization({
      name: `${namePrefix}${layerId}ln2`
    }).apply(x2) as tf.SymbolicTensor;
    x3 = mlp(x3, {
      layerUnits,
      dropout,
      namePrefix: `${namePrefix}${layerId}mlp`
    });
    network = tf.layers.add(
    ).apply([ x3, x2 ]) as tf.SymbolicTensor;
  }

  network = tf.layers.layerNormalization({
    name: `${namePrefix}ln`
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.dense({
    units: headDim,
    name: `${namePrefix}proj`
  }).apply(network) as tf.SymbolicTensor;
  network = tf.layers.flatten(
  ).apply(network) as tf.SymbolicTensor;
  network = tf.layers.dropout({
    rate: headDropout
  }).apply(network) as tf.SymbolicTensor;
  network = mlp(network, {
    layerUnits: headMlpUnits,
    dropout: headDropout,
    namePrefix: `${namePrefix}mlp`
  });
  return network;
};


export {
  convLayer2D,
  residualNetwork2D,
  copyWeights,
  countResidualLayers,
  denseLayer,
  kld,
  transformer
}
