import * as tf from '@tensorflow/tfjs';
import { Input, Output } from '../interfaces/game-rules';
import {
  saveModel,
  loadModel,
  getTrainDir,
  loadTrainLosses,
  loadTrainModel,
  saveTrainModel,
  saveTrainLosses
} from '../lib/api';
import { copyWeights } from './networks';

const trainOrder = [
  'policy',
  'reward'
];
const ensembleSize = 4;
const validationSplit = 0.01;

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

const randomAugment = (input: Input) => {
  const isColorDrop = Math.random() < 0.5;
  const isPositionDrop = !isColorDrop;

  const height = input.length;
  const width = input[0].length;
  const depth = input[0][0].length;

  const iDrop = isPositionDrop ?
    Math.floor(Math.random() * height) : -1;
  const jDrop = isPositionDrop ?
    Math.floor(Math.random() * width) : -1;
  const zDrop = isColorDrop ?
    Math.floor(Math.random() * depth) : -1;

  return input.map(
    (row, i) => row.map(
      (column, j) => column.map(
        (value, z) => isColorDrop ?
          (z === zDrop) ?
            0 : value
          :
          (i === iDrop && j === jDrop) ?
            0 : value
      )
    )
  ) as Input;
};
export interface AlphaNetworkOptions {
  height: number;
  width: number;
  depth: number;
  graph: TfGraph;
  gameName: string;
};

export default class AlphaNetwork {
  private height: number;
  private width: number;
  private depth: number;
  private graph: TfGraph;
  private model: tf.LayersModel;
  private gameName: string;
  constructor(options: AlphaNetworkOptions) {
    this.height = options.height;
    this.width = options.width;
    this.depth = options.depth;
    this.graph = options.graph;
    this.gameName = options.gameName;
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
    const valSplit = Math.floor(inputs.length * validationSplit);
    const trainInputs = inputs.slice(0, -valSplit);
    const valInputs = inputs.slice(-valSplit);
    const trainOutputs = outputs.slice(0, -valSplit);
    const valOutputs = outputs.slice(-valSplit);

    const trainDir = await getTrainDir(this.gameName);
    let trainLosses = [] as number[][];
    if (trainDir.includes('losses')) {
      trainLosses = await loadTrainLosses(this.gameName);
    }
    let trainModel = this.model;
    if (trainDir.includes('model')) {
      trainModel = await loadTrainModel(this.gameName);
    }

    for (let epoch = 0; epoch < this.graph.epochs; epoch++) {
      if (!trainLosses[epoch]) {
        trainLosses[epoch] = [];
      }
      for (let head = 0; head < ensembleSize; head++) {
        let headLoss = trainLosses[epoch][head];
        if (headLoss) {
          continue;
        }
        const headLosses = [] as number[];
        const headTrainOrder = trainOrder.slice();
        if (head % 2 === 1) headTrainOrder.reverse();
        for (let task of headTrainOrder) {
          console.log(`training epoch${epoch + 1} ${task}${head + 1} ...`);
          const xsTensor = tf.tensor4d(trainInputs.map(randomAugment));
          const ysTensor = tf.tensor2d(trainOutputs.map(
            output => task === 'policy' ?
              output[0] : [output[1]]
          ));
          const valXsTensor = tf.tensor4d(valInputs);
          const valYsTensor = tf.tensor2d(valOutputs.map(
            output => task === 'policy' ?
              output[0] : [output[1]]
          ));
          const input = tf.input({
            shape: [this.height, this.width, this.depth]
          });
          const commonNetwork = this.graph.createCommon(head + 1)(input);
          const headNetwork = task === 'policy' ?
            this.graph.createPolicy(head + 1)(commonNetwork) :
            this.graph.createReward(head + 1)(commonNetwork);
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
          if (
            !trainLosses[epoch - 1] ||
            !trainLosses[epoch - 1][head]
          ) {
            copyWeights(this.model, taskModel);
          } else {
            copyWeights(trainModel, taskModel);
          }
          const epochs = task === 'policy' ? 2 : 1;

          const taskHistory = await taskModel.fit(
            xsTensor,
            ysTensor,
            {
              batchSize: this.graph.batchSize,
              epochs,
              shuffle: true,
              validationData: [
                valXsTensor,
                valYsTensor
              ]
            }
          );
          xsTensor.dispose();
          ysTensor.dispose();
          valXsTensor.dispose();
          valYsTensor.dispose();
          const taskLoss = taskHistory.history.val_loss[
            epochs - 1
          ] as number;
          headLosses.push(taskLoss);
          copyWeights(taskModel, this.model);
          copyWeights(taskModel, trainModel);
        }
        headLoss = headLosses.reduce(
          (total, current) => total + current,
          0
        );
        trainLosses[epoch][head] = +headLoss.toPrecision(3);
        await saveTrainModel(trainModel, this.gameName);
        await saveTrainLosses(this.gameName, trainLosses);
      }
    }

    const loss = trainLosses[trainLosses.length - 1]
      .reduce((total, current) => total + current, 0) / ensembleSize;
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
