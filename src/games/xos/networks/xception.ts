import * as tf from '@tensorflow/tfjs';
import { TfGraph } from '../../../lib/alpha-network';
import {
    kld,
    xception
} from '../../../lib/networks';

// const numFilters = 64;
const numLayers = 6;
const numFrozen = 1;
const batchSize = 1024;
const epochs = 10;
const learningRate = 0.0005;
// const dropout = 0.1;

interface TfNetworkOptions {
  height: number;
  width: number;
  depth: number;
}

export default class Xception implements TfGraph {
  private height: number;
  private width: number;
  private depth: number;
  batchSize = batchSize;
  epochs = epochs;
  constructor(options: TfNetworkOptions) {
    this.height = options.height;
    this.width = options.width;
    this.depth = options.depth;
  }
  createCommon(id?: number) {
    const name = id ?
      `common${id}` : 'common';
    return (input: tf.SymbolicTensor) => {
      let common = xception(input, {
        numLayers,
        name: `${name}Xception`,
        numFrozen
      });
      return common;
    };
  }
  createPolicy(id?: number) {
    const name = id ?
      `policy${id}` : 'policy';
    return (common: tf.SymbolicTensor) => {
      let policy = tf.layers.separableConv2d({
        filters: 2,
        kernelSize: 3,
        strides: 1,
        padding: 'same',
        name: `${name}decConv`,
      }).apply(common) as tf.SymbolicTensor;
      policy = tf.layers.batchNormalization({
        name: `${name}decBn`,
      }).apply(policy) as tf.SymbolicTensor;
      policy = tf.layers.activation({
        activation: 'relu'
      }).apply(policy) as tf.SymbolicTensor;
  
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
  createReward(id?: number) {
    const name = id ?
      `reward${id}` : 'reward';
    return (common: tf.SymbolicTensor) => {
      let reward = tf.layers.separableConv2d({
        filters: 2,
        kernelSize: 3,
        strides: 1,
        padding: 'same',
        name: `${name}decConv`,
      }).apply(common) as tf.SymbolicTensor;
      reward = tf.layers.batchNormalization({
        name: `${name}decBn`,
      }).apply(reward) as tf.SymbolicTensor;
      reward = tf.layers.activation({
        activation: 'relu'
      }).apply(reward) as tf.SymbolicTensor;
  
      reward = tf.layers.flatten(
      ).apply(reward) as tf.SymbolicTensor;
    
      // reward = tf.layers.dropout({
      //   rate: 0.5
      // }).apply(reward) as tf.SymbolicTensor;

      reward = tf.layers.dense({
        units: 1,
        name: `${name}Head`,
        activation: 'tanh'
      }).apply(reward) as tf.SymbolicTensor;
      return reward;
    };
  }
  policyCompileArgsCreator() {
    return {
      optimizer: tf.train.adam(learningRate),
      loss: kld
    }
  }
  rewardCompileArgsCreator() {
    return {
      optimizer: tf.train.adam(learningRate),
      loss: tf.losses.meanSquaredError
    }
  }
};