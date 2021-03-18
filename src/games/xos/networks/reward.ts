import * as tf from '@tensorflow/tfjs';
import { TfNetwork } from '../../../lib/alpha-network';
import {
    residualNetwork2D,
    denseLayer,
    convLayer2D
} from '../../../lib/networks';

const numFilters = 64;
const numLayers = 5;
const batchSize = 1024;
const epochs = 10;
const learningRate = 0.05;
const dropout = 0.1;

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
    // const optimizer = tf.train.adam(learningRate);
    const optimizer = tf.train.sgd(learningRate);
    this.compileArgs = {
      optimizer: optimizer,
      loss: tf.losses.meanSquaredError
    };
  }
  createGraph(id?: number) {
    const namePrefix = id ?
      `reward${id}` : 'reward';
    return (input: tf.SymbolicTensor) => {
      let network = residualNetwork2D(input, {
        numLayers,
        numFilters,
        kernelSize: 3,
        namePrefix,
        dropout: dropout
      });
    
      let reward = convLayer2D(network, {
        name: `${namePrefix}Conv`,
        numFilters: 2,
        kernelSize: 1,
        padding: 'same',
        dropout: dropout
      });
    
      reward = tf.layers.flatten(
      ).apply(reward) as tf.SymbolicTensor;
    
      reward = denseLayer(reward, {
        name: `${namePrefix}Dense`,
        units: this.height * this.width,
        dropout
      });
  
      reward = denseLayer(reward, {
        name: `${namePrefix}DenseHead`,
        units: 1,
        dropout,
        noActivation: true
      });
    
      reward = tf.layers.activation({
        activation: 'tanh',
        name: namePrefix
      }).apply(reward) as tf.SymbolicTensor;
      return reward;
    };
  }
};