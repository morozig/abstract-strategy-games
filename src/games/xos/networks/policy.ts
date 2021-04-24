import * as tf from '@tensorflow/tfjs';
// import { TfGraph } from '../../../lib/alpha-network';
import {
    kld,
    xception
} from '../../../lib/networks';

// const numFilters = 64;
const numLayers = 5;
const batchSize = 1024;
const epochs = 15;
const learningRate = 0.001;
// const dropout = 0.1;

interface TfNetworkOptions {
  height: number;
  width: number;
  depth: number;
}

export default class Policy {
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
  createCommon(id?: number) {
    const name = id ?
      `policy${id}` : 'policy';
    return (input: tf.SymbolicTensor) => {
      let policy = xception(input, {
        numLayers,
        name: `${name}Xception`
      });
  
      policy = tf.layers.flatten(
      ).apply(policy) as tf.SymbolicTensor;
    
      // policy = tf.layers.dropout({
      //   rate: 0.5
      // }).apply(policy) as tf.SymbolicTensor;

      policy = tf.layers.dense({
        units: this.height * this.width,
        name: `${name}Head`,
        activation: 'softmax'
      }).apply(policy) as tf.SymbolicTensor;
      return policy;
    };
  }
};