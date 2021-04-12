import * as tf from '@tensorflow/tfjs';
import { TfNetwork } from '../../../lib/alpha-network';
import {
    transformer
} from '../../../lib/networks';

const dim = 32;
const numHeads = 2;
const mlpUnits = [ dim ];
const numLayers = 5;
const dropout = 0.1;
const headDim = 2;
const headMlpUnits = [ 128, 64 ];
const headDropout = 0.5;
const batchSize = 1024;
const epochs = 10;
const learningRate = 0.05;

interface TfNetworkOptions {
  height: number;
  width: number;
  depth: number;
}

export default class Reward implements TfNetwork {
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
      loss: tf.losses.meanSquaredError
    };
  }
  createGraph(id?: number) {
    const namePrefix = id ?
      `reward${id}` : 'reward';
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

      let reward = tf.layers.dense({
        units: 1,
        activation: 'tanh'
      }).apply(network) as tf.SymbolicTensor;
      return reward;
    };
  }
};