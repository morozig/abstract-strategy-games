import * as tf from '@tensorflow/tfjs';
import { TfNetwork } from '../../../lib/alpha-network';
import {
    kld,
    transformer
} from '../../../lib/networks';

const dim = 64;
const numHeads = 4;
const mlpUnits = [ dim * 2, dim ];
const numLayers = 5;
const dropout = 0.1;
const headDim = 2;
const headMlpUnits = [ 1024, 512 ];
const headDropout = 0.5;
const batchSize = 64;
const epochs = 15;
const learningRate = 0.05;

interface TfNetworkOptions {
  height: number;
  width: number;
  depth: number;
}

export default class Policy implements TfNetwork {
  private height: number;
  private width: number;
  private depth: number;
  batchSize = batchSize;
  epochs = epochs;
  compileArgs: tf.ModelCompileArgs;
  constructor(options: TfNetworkOptions) {
    this.height = options.height;
    this.width = options.width;
    this.depth = options.depth;
    const optimizer = tf.train.adam(learningRate);
    // const optimizer = tf.train.sgd(learningRate);
    this.compileArgs = {
      optimizer: optimizer,
      loss: kld
    };
  }
  createGraph(id?: number) {
    const namePrefix = id ?
      `policy${id}` : 'policy';
    const numTiles = this.width * this.height;
    return (input: tf.SymbolicTensor) => {
      let network = transformer(input, {
        dim,
        numHeads,
        numTiles,
        mlpUnits,
        numLayers,
        dropout,
        headDim,
        headMlpUnits,
        headDropout,
        namePrefix
      });

      let policy = tf.layers.dense({
        units: numTiles,
        activation: 'softmax'
      }).apply(network) as tf.SymbolicTensor;
      return policy;
    };
  }
};