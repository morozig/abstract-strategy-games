import GameHistory from '../interfaces/game-history';
import GameRules, {
  Input,
  Output,
  Pair
} from '../interfaces/game-rules';
import AlphaNetwork, { TypedInput } from './alpha-network';
import * as tf from '@tensorflow/tfjs';

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
    const statePairs = new Map<string, Pair[]>();

    for (let gameHistory of gameHistories) {
      const pairs = this.rules.getPairs(gameHistory);
      for (let pair of pairs) {
        const { input } = pair;
        const hash = input.flat(2).join('');
        if (!statePairs.has(hash)) {
          statePairs.set(hash, [pair]);
        } else {
          const currentPairs = statePairs.get(hash);
          if (currentPairs) {
            statePairs.set(hash, currentPairs.concat(pair));
          }
        }
      }
    }
    const pairs = Array.from(statePairs.values())
      .map(currentPairs => {
        const count = currentPairs.length;
        const pair = currentPairs[0];
        if (count <= 1) {
          return pair;
        } else {
          const totalReward = currentPairs.reduce(
            (total, current) => total + current.output[1],
            0
          );
          pair.output[1] = totalReward / count;
          return pair;
        }
      });
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
  predictBatch(inputs: TypedInput[]){
    return this.network.predict(inputs);
  }
};
