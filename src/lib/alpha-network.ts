import * as tf from '@tensorflow/tfjs';
import { Input, Output } from '../interfaces/game-rules';
import {
  saveModel,
  loadModel
} from '../lib/api';

export type TypedInput = Float32Array;
export type TypedOutput = [Float32Array, Float32Array];

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
    inputs: Input[],
    outputs: Output[]
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

  async predict(inputs: Input[]):
    Promise<Output[]>;
  async predict(inputs: TypedInput[]):
    Promise<TypedOutput[]>;
  async predict(inputs: Input[] | TypedInput[]) {
    const inputsTensor = tf.tensor(
      inputs,
      [
        inputs.length,
        this.height,
        this.width,
        this.depth
      ]
    ) as tf.Tensor4D;

    const [ policiesTensor, rewardsTensor ] = this.model.predict(
      inputsTensor
    ) as [tf.Tensor2D, tf.Tensor2D];
    const isTyped = inputs[0] instanceof Float32Array;

    if (!isTyped) {
      const policies = await policiesTensor.array();
      const rewards = await rewardsTensor.array();
  
      inputsTensor.dispose();
      policiesTensor.dispose();
      rewardsTensor.dispose();
  
      const outputs = policies.map(
        (policy, i) => [policy, rewards[i][0]] as Output
      );
      return outputs;
    } else {
      const policies = await policiesTensor.data() as Float32Array;
      const rewards = await rewardsTensor.data() as Float32Array;

      inputsTensor.dispose();
      policiesTensor.dispose();
      rewardsTensor.dispose();

      const policySize = this.height * this.width;
      const rewardSize = 1;

      const outputs = (inputs as TypedInput[]).map(
        (_, i) => [
          policies.subarray(i * policySize, (i + 1) * policySize),
          rewards.subarray(i * rewardSize, (i + 1) * rewardSize),
        ] as TypedOutput
      );
      return outputs;
    }
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
