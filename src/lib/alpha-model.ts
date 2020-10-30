import GameHistory from '../interfaces/game-history';
import Batcher from '../lib/batcher';
import PolicyAction from '../interfaces/policy-action';
import GameRules from '../interfaces/game-rules';
import { PlaneSymmetry, plane } from '../lib/transforms';
import { softMax } from '../lib/helpers';
import AlphaNetwork from './alpha-network';
import * as tf from '@tensorflow/tfjs';

type Input = number[][][];
type Output = [number[], number];

interface Pair {
  input: Input;
  output: Output;
};

interface AlphaModelOptions {
  gameName: string;
  rules: GameRules,
  model: tf.LayersModel;
  batchSize: number;
  epochs: number;
  learningRate: number;
};

export default class AlphaModel {
  private gameName: string;
  private rules: GameRules;
  private model: tf.LayersModel;
  private batchSize: number;
  private epochs: number;
  private learningRate: number;
  private network: AlphaNetwork;
  constructor(options: AlphaModelOptions) {
    this.gameName = options.gameName;
    this.rules = options.rules;
    this.model = options.model;
    this.batchSize = options.batchSize;
    this.epochs = options.epochs;
    this.learningRate = options.learningRate;
    this.network = new AlphaNetwork({
      height: this.rules.height,
      width: this.rules.width,
      depth: this.rules.depth,
      model: this.model,
      batchSize: this.batchSize,
      epochs: this.epochs,
      learningRate: this.learningRate
    });
  }
  async train(
    gameHistories: GameHistory[]
  ) {
    const inputs = [] as Input[];
    const outputs = [] as Output[];
    const uniquePairs = new Map<string, Pair>();

    for (let gameHistory of gameHistories) {
      const pairs = this.rules.getPairs(gameHistory);
      for (let pair of pairs) {
        const { input } = pair;
        const hash = input.flat(2).join('');
        if (!uniquePairs.has(hash)) {
          uniquePairs.set(hash, pair);
        }
      }
    }
    const pairs = Array.from(uniquePairs.values());
    pairs.sort(() => Math.random() - 0.5);

    for (let { input, output } of pairs) {
      inputs.push(input);
      outputs.push(output);
    }
    console.log(`training data length: ${inputs.length}`);
    const loss = await this.network.fit(inputs, outputs);
    return loss;
  }
  async save(name: string){
      await this.network.save(this.gameName, name);
  }
  async load(name: string){
      await this.network.load(this.gameName, name);
  }
  async predict(history: number[]) {
    const input = this.rules.getInput(history);
    const [output] = await this.network.predict([input]);

    const [ policy, reward ] = output;
    return {
      reward,
      policy
    };
  }
  predictBatches(batches: Float32Array[]){
    return this.network.predictBatches(batches);
  }
};
