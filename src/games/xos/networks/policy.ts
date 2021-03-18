import * as tf from '@tensorflow/tfjs';
import { TfNetwork } from '../../../lib/alpha-network';
import {
    residualNetwork2D,
    denseLayer,
    convLayer2D,
    kld
} from '../../../lib/networks';

const numFilters = 64;
const numLayers = 5;
const batchSize = 1024;
const epochs = 15;
const learningRate = 0.05;
const dropout = 0.1;

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
    // const optimizer = tf.train.adam(learningRate);
    const optimizer = tf.train.sgd(learningRate);
    this.compileArgs = {
      optimizer: optimizer,
      loss: kld
    };
  }
  createGraph(id?: number) {
    const namePrefix = id ?
      `policy${id}` : 'policy';
    return (input: tf.SymbolicTensor) => {
      let network = residualNetwork2D(input, {
        numLayers,
        numFilters,
        kernelSize: 3,
        namePrefix,
        dropout: dropout
      });
    
      let policy = convLayer2D(network, {
        name: `${namePrefix}Conv`,
        numFilters: 2,
        kernelSize: 1,
        padding: 'same',
        dropout: dropout
      })
    
      policy = tf.layers.flatten(
      ).apply(policy) as tf.SymbolicTensor;
    
      policy = denseLayer(policy, {
        name: `${namePrefix}Dense`,
        units: 2 * this.height * this.width,
        dropout
      });
  
      policy = denseLayer(policy, {
        name: `${namePrefix}DenseHead`,
        units: this.height * this.width,
        dropout,
        noActivation: true
      });
    
      policy = tf.layers.activation({
        activation: 'softmax',
        name: namePrefix
      }).apply(policy) as tf.SymbolicTensor;
      return policy;
    };
  }
};