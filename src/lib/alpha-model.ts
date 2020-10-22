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
    const pairs = [] as Pair[];

    for (let gameHistory of gameHistories) {
      const symHistories = this.rules.getSymHistories(
        gameHistory.history
      );
      for (let entries of symHistories) {
          const history = entries.map(({action}) => action);
          const states = getStates(history, this.rules);
          states.pop();
          for (let i = 0; i < states.length; i++) {
              const layerStates = states.slice(0, i + 1);
              const { input } = getInput(layerStates, this.rules);
              const lastState = states[i];
              const lastPlayerIndex = lastState.playerIndex;
              const reward = gameHistory.rewards[lastPlayerIndex];
              const { policy } = gameHistory.history[i];
              const output = getOutput(reward, policy);
              pairs.push({
                  input,
                  output
              });
          }
      }
    }
    pairs.forEach(pair => {
        inputs.push(pair.input);
        outputs.push(pair.output)
    });
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
      const states = getStates(history, this.rules);
      const { input } = getInput(states, this.rules);
      let output: Output;
      if (!this.batcher) {
          [output] = await this.network.predict([input]);
      } else {
          output = await this.batcher.call(input);
      }
      const [ policy, reward ] = output;
      return {
          reward,
          policy
      };
  }
};
