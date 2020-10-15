import * as tf from '@tensorflow/tfjs';
import {
  saveModel,
  loadModel
} from './api';

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
    inputs: number[][][][],
    outputs: [number[], number][]
  ){
    const xsTensor = tf.tensor4d(inputs);
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
  }
  async predict(inputs: number[][][][]) {
    const inputsTensor = tf.tensor4d(inputs);
    const [ policiesTensor, rewardsTensor ] = this.model.predict(
      inputsTensor
    ) as [tf.Tensor2D, tf.Tensor2D];
    const policies = await policiesTensor.array();
    const rewards = await rewardsTensor.array();

    inputsTensor.dispose();
    policiesTensor.dispose();
    rewardsTensor.dispose();

    const outputs = policies.map(
      (policy, i) => [policy, rewards[i][0]] as [number[], number]
    );
    return outputs;
  }
  async predictBatches(batches: Float32Array[]) {
    const inputSize = this.height * this.width * this.depth;
    const { batchesIndices } = batches.reduce(
      ({batchesIndices, last}, current) => ({
        batchesIndices: batchesIndices.concat(
          last + current.length / inputSize
        ),
        last: last + current.length / inputSize
      }),
      {batchesIndices: [0], last: 0}
    );
    const inputsLength = batchesIndices[
      batchesIndices.length - 1
    ];
    const inputsTensor = tf.tensor(
      batches,
      [
        inputsLength,
        this.height,
        this.width,
        this.depth
      ]
    ) as tf.Tensor4D;
    const [ policiesTensor, rewardsTensor ] = this.model.predict(
      inputsTensor
    ) as [tf.Tensor2D, tf.Tensor2D];

    const policies = await policiesTensor.data() as Float32Array;
    const rewards = await rewardsTensor.data() as Float32Array;

    inputsTensor.dispose();
    policiesTensor.dispose();
    rewardsTensor.dispose();

    const policySize = this.height * this.width;
    const policyBatches = batchesIndices.map(
      (batchIndex, i, arr) => policies.subarray(
        batchIndex * policySize,
        arr[i + 1] * policySize
      )
    );

    const rewardSize = 1;
    const rewardBatches = batchesIndices.map(
      (batchIndex, i, arr) => rewards.subarray(
        batchIndex * rewardSize,
        arr[i + 1] * rewardSize
      )
    );

    const outputs = [
      policyBatches, rewardBatches
    ] as [Float32Array[], Float32Array[]];

    return outputs;
  }
  async save(gameName: string, modelName: string) {
    await saveModel(
      this.model,
      gameName,
      modelName
    );
  }
  async load(gameName: string, modelName: string) {
    this.model.dispose();
    this.model = await loadModel(gameName, modelName);
    this.compile();
  }
};
