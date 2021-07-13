import GameHistory from '../interfaces/game-history';
import GameRules, {
  Input,
  Output,
  Pair
} from '../interfaces/game-rules';
import AlphaNetwork, { TfGraph, TypedInput } from './alpha-network';
import {
  loadTrainExamples,
  getTrainDir,
  saveTrainExamples
} from '../lib/api';
import { retry } from './helpers';

interface AlphaModelOptions {
  gameName: string;
  rules: GameRules,
  graph: TfGraph;
};

export default class AlphaModel {
  readonly gameName: string;
  private rules: GameRules;
  private graph: TfGraph;
  private network: AlphaNetwork;
  constructor(options: AlphaModelOptions) {
    this.gameName = options.gameName;
    this.rules = options.rules;
    this.graph = options.graph;
    this.network = new AlphaNetwork({
      height: this.rules.height,
      width: this.rules.width,
      depth: this.rules.depth,
      graph: this.graph
    });
  }
  async train(
    gameHistories: GameHistory[],
    modelName: string
  ) {
    let inputs = [] as Input[];
    let outputs = [] as Output[];
    const statePairs = new Map<string, Pair[]>();

    const trainDir = await getTrainDir(this.gameName, modelName);
    if (trainDir.includes('examples')){
      const trainExamples = await loadTrainExamples(this.gameName, modelName);
      inputs = trainExamples.inputs;
      outputs = trainExamples.outputs;
    } else {
      console.log('preparing training examples...');
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
      await saveTrainExamples(
        this.gameName,
        modelName,
        {
          inputs,
          outputs
        }
      );
    }

    console.log(`training data length: ${inputs.length}`);
    const loss = await this.network.fit({
      gameName: this.gameName,
      modelName,
      inputs,
      outputs
    });
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
    const [output] = await retry(() => this.network.predict([input]));

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
