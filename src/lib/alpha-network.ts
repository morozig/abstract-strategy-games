import * as tf from '@tensorflow/tfjs';
import { Input, Output } from '../interfaces/game-rules';
import {
  saveModel,
  loadModel
} from '../lib/api';
import { copyWeights } from './networks';

const trainOrder = [
  'policy',
  'reward'
];
const ensembleSize = 4;

export type TypedInput = Float32Array;
export type TypedOutput = [Float32Array, Float32Array];

export type GraphCreator = (id?: number) =>
  (input: tf.SymbolicTensor) => tf.SymbolicTensor;

export type CompileArgsCreator = () => tf.ModelCompileArgs;
export interface TfGraph {
  readonly createCommon: GraphCreator;
  readonly createPolicy: GraphCreator;
  readonly createReward: GraphCreator;
  readonly policyCompileArgsCreator: CompileArgsCreator;
  readonly rewardCompileArgsCreator: CompileArgsCreator;
  readonly batchSize: number;
  readonly epochs: number;
};

export interface AlphaNetworkOptions {
  height: number;
  width: number;
  depth: number;
  graph: TfGraph;
};

export default class AlphaNetwork {
  private height: number;
  private width: number;
  private depth: number;
  private graph: TfGraph;
  private model: tf.LayersModel;
  constructor(options: AlphaNetworkOptions) {
    this.height = options.height;
    this.width = options.width;
    this.depth = options.depth;
    this.graph = options.graph;
    const input = tf.input({
      shape: [this.height, this.width, this.depth]
    });
    const commonNetworks = new Array(ensembleSize)
      .fill(undefined)
      .map((_, i) => this.graph.createCommon(i + 1)(input));
    const policyHeads = commonNetworks
      .map((network, i) => this.graph.createPolicy(i + 1)(network));
    const policyAverage = tf.layers.average({
      name: 'policy'
    }).apply(policyHeads) as tf.SymbolicTensor;

    const rewardHeads = commonNetworks
      .map((network, i) => this.graph.createReward(i + 1)(network));
    const rewardAverage = tf.layers.average({
      name: 'reward'
    }).apply(rewardHeads) as tf.SymbolicTensor;

    this.model = tf.model(
      {
        inputs: input,
        outputs: [
          policyAverage,
          rewardAverage
        ]
      }
    );
  }
  async fit(
    inputs: Input[],
    outputs: Output[]
  ){
    const xsTensor = tf.tensor4d(inputs);
    const losses = [] as number[];

    for (let i = 1; i <= ensembleSize; i++) {
      const epochsLosses = [] as number[];
      for (let j = 1; j <= this.graph.epochs; j++) {
        for (let task of trainOrder) {
          console.log(`training ${task}${i} epoch${j}...`);
          const ysTensor = tf.tensor2d(outputs.map(
            output => task === 'policy' ?
              output[0] : [output[1]]
          ));
          const input = tf.input({
            shape: [this.height, this.width, this.depth]
          });
          const commonNetwork = this.graph.createCommon(i + 1)(input);
          const headNetwork = task === 'policy' ?
            this.graph.createPolicy(i + 1)(commonNetwork) :
            this.graph.createReward(i + 1)(commonNetwork);
          const taskModel = tf.model(
            {
              inputs: input,
              outputs: headNetwork
            }
          );
          taskModel.compile(task === 'policy' ?
            this.graph.policyCompileArgsCreator() :
            this.graph.rewardCompileArgsCreator()
          );
          copyWeights(this.model, taskModel);
          
          const taskHistory = await taskModel.fit(
            xsTensor,
            ysTensor,
            {
              batchSize: this.graph.batchSize,
              epochs: 1,
              shuffle: true,
              validationSplit: 0.01
            }
          );
          ysTensor.dispose();
          const taskLoss = taskHistory.history.val_loss[
            0
          ] as number;
          epochsLosses.push(taskLoss);
          copyWeights(taskModel, this.model);
        }
      }
      losses.push(
        epochsLosses
        .slice(-2)
        .reduce(
          (total, current) => total + current, 0
        ) / ensembleSize
      )
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
