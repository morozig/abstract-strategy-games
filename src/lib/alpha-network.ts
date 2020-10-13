import * as tf from '@tensorflow/tfjs';
import { TypedArray } from '@tensorflow/tfjs';

export interface AlphaNetworkOptions {
  height: number;
  width: number;
  depth: number;
  batchSize: number;
  epochs: number;
  learningRate: number;
  model: tf.LayersModel;
};

export default class AlphaNetwork {
  private height: number;
  private width: number;
  private depth: number;
  private batchSize: number;
  private epochs: number;
  private learningRate: number;
  private model: tf.LayersModel;
  constructor(options: AlphaNetworkOptions) {
    this.height = options.height;
    this.width = options.width;
    this.depth = options.depth;
    this.batchSize = options.batchSize;
    this.epochs = options.epochs;
    this.learningRate = options.learningRate;
    this.model = options.model;
    this.compile();
  }
  private compile() {
    const optimizer = tf.train.adam(this.learningRate);

    this.model.compile({
      optimizer: optimizer,
      loss: [
          'categoricalCrossentropy',
          'meanSquaredError'
      ],
      metrics: ['accuracy']
    });
  }
  async fit(inputs: TypedArray[], )
};
