import PolicyAgent from '../interfaces/policy-agent';
import GameRules from '../interfaces/game-rules';
import Mcts from './mcts';
import AlphaModel from '../lib/alpha-model';

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
    return {
      reward,
      policy
    };
  };
};

export default class Alpha implements PolicyAgent {
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
  policyAct() {
    return this.mcts.policyAct();
  }
  init() {
    return this.mcts.init();
  }
  step(action: number) {
    return this.mcts.step(action);
  }
};