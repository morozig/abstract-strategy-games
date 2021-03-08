import * as tf from '@tensorflow/tfjs';
import { TfNetwork } from '../../../lib/alpha-network';
import {
    residualNetwork2D,
    denseLayer,
    convLayer2D,
    kld
} from '../../../lib/networks';

const numFilters = 128;
const numLayers = 5;
const batchSize = 1024;
const epochs = 10;
const learningRate = 0.01;
const dropout = 0.3;

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
    this.compileArgs = {
      optimizer: optimizer,
      loss: kld
    };
  }
  graph(input: tf.SymbolicTensor) {
    let network = residualNetwork2D(input, {
      numLayers,
      numFilters,
      kernelSize: 3,
      namePrefix: 'policy'
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
      units: 2 * this.height * this.width,
      dropout
    });
  
    policy = tf.layers.dense({
      name: 'policyDenseHead',
      units: this.height * this.width
    }).apply(policy) as tf.SymbolicTensor;
  
    policy = tf.layers.activation({
      activation: 'softmax',
      name: 'policy'
    }).apply(policy) as tf.SymbolicTensor;
    return policy;
  }
};