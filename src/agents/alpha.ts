import HistoryAgent from '../interfaces/history-agent';
import GameRules from '../interfaces/game-rules';
import Mcts from './mcts';
import AlphaModel from '../lib/alpha-model';
import { softMax } from '../lib/helpers';

const predictionTemp = 0.5;
interface AlphaOptions {
  gameRules: GameRules;
  model: AlphaModel;
  modelName?: string;
  planCount?: number;
  randomize?: boolean;
}

const modelPredictor = (model: AlphaModel, name?: string) => {
  let modelLoaded = !name;
  return async(history: number[]) => {
    if (!modelLoaded && name) {
      await model.load(name);
      modelLoaded = true;
    }
    const { reward, policy } = await model.predict(history);
    const softPolicy = softMax(policy, predictionTemp);

    return {
      reward,
      policy: softPolicy
    };
  };
};

export default class Alpha implements HistoryAgent {
  private mcts: Mcts;
  constructor(options: AlphaOptions) {
    this.mcts = new Mcts({
      gameRules: options.gameRules,
      predict: modelPredictor(options.model, options.modelName),
      planCount: options.planCount,
      randomize: options.randomize
    });
  }
  act() {
    return this.mcts.act();
  }
  historyAct() {
    return this.mcts.historyAct();
  }
  init() {
    return this.mcts.init();
  }
  step(action: number) {
    return this.mcts.step(action);
  }
};