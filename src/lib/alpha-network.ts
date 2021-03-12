import * as tf from '@tensorflow/tfjs';
import { Input, Output } from '../interfaces/game-rules';
import {
  saveModel,
  loadModel
} from '../lib/api';
import { copyWeights } from './networks';

const trainOrder = [
  'reward',
  'policy',
];
// const trainOrder = [
//   'policy',
//   'reward'
// ];

export type TypedInput = Float32Array;
export type TypedOutput = [Float32Array, Float32Array];

export interface TfNetwork {
  readonly graph: (input: tf.SymbolicTensor) => tf.SymbolicTensor;
  readonly batchSize: number;
  readonly epochs: number;
  readonly compileArgs: tf.ModelCompileArgs;
};

export interface AlphaNetworkOptions {
  height: number;
  width: number;
  depth: number;
  policy: TfNetwork;
  reward: TfNetwork;
};

export default class AlphaNetwork {
  private height: number;
  private width: number;
  private depth: number;
  private policy: TfNetwork;
  private reward: TfNetwork;
  private model: tf.LayersModel;
  constructor(options: AlphaNetworkOptions) {
    this.height = options.height;
    this.width = options.width;
    this.depth = options.depth;
    this.policy = options.policy;
    this.reward = options.reward;
    const input = tf.input({
      shape: [this.height, this.width, this.depth]
    });
    this.model = tf.model(
      {
        inputs: input,
        outputs: [
          this.policy.graph(input),
          this.reward.graph(input)
        ]
      }
    );
  }
  async fit(
    inputs: Input[],
    outputs: Output[]
  ){
    const xsTensor = tf.tensor4d(inputs);
    const input = tf.input({
      shape: [this.height, this.width, this.depth]
    });
    const losses = [] as number[];

    for (let head of trainOrder) {
      console.log(`training ${head} model...`);
      const headNetwork = head === 'policy' ?
        this.policy : this.reward;
      const ysTensor = tf.tensor2d(outputs.map(
        output => head === 'policy' ?
          output[0] : [output[1]]
      ));
      const headModel = tf.model(
        {
          inputs: input,
          outputs: headNetwork.graph(input)
        }
      );
      copyWeights(this.model, headModel);
      headModel.compile(headNetwork.compileArgs);
      const headHistory = await headModel.fit(
        xsTensor,
        ysTensor,
        {
          batchSize: headNetwork.batchSize,
          epochs: headNetwork.epochs,
          shuffle: true,
          validationSplit: 0.01
        }
      );
      ysTensor.dispose();
      const headLoss = headHistory.history.val_loss[
        headNetwork.epochs - 1
      ] as number;
      losses.push(headLoss);
      copyWeights(headModel, this.model);
    }

    xsTensor.dispose();
    const loss = losses.reduce((total, current) => total + current, 0);
    console.log('loss:', loss.toPrecision(3));
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
          policies.slice(i * policySize, (i + 1) * policySize),
          rewards.slice(i * rewardSize, (i + 1) * rewardSize),
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
  }
};
