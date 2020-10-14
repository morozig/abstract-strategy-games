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
  async fit(
    inputs: Float32Array[],
    outputs: [
      Float32Array[],
      Float32Array[]
    ]
  ){
    const inputLength = this.height * this.width * this.depth;
    const totalLength = inputs.reduce(
      (total, current) => total + current.length,
      0
    );
    const totalInputs = totalLength / inputLength;
    const xsTensor = tf.tensor(inputs, [
      totalInputs,
      this.height,
      this.width,
      this.depth
    ]);
    const policiesTensor = tf.tensor2d(outputs.map(
      output => output[0]
    ));
    const rewardsTensor = tf.tensor2d(outputs.map(
      output => [output[1]]
    ));
    const ysTensors = [
      policiesTensor,
      rewardsTensor
    ];

    const trainingHistory = await this.model.fit(
      xsTensor,
      ysTensors,
      {
        batchSize: this.batchSize,
        epochs: this.epochs,
        shuffle: true,
        validationSplit: 0.01,
        callbacks: {
          onEpochEnd: console.log
        }
      }
    );

    xsTensor.dispose();
    policiesTensor.dispose();
    rewardsTensor.dispose();
    console.log(trainingHistory);
    const loss = trainingHistory.history.loss[
      this.epochs - 1
    ] as number;
    return loss;
  };

  async predict(inputs: Float32Array[]) {

  }
};
