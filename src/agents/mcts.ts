import GameRules from '../interfaces/game-rules';
import GamePrediction from '../interfaces/game-prediction';
import {
  randomOf,
  indexMax,
  indexSoftMax
} from '../lib/helpers';
import PolicyAgent from '../interfaces/policy-agent';
import GameState from '../interfaces/game-state';

const cPuct = 5;
const finishTemp = 0.1;
const printPolicy = false;

export type Predictor = (history: number[]) => Promise<GamePrediction>;

interface MctsOptions {
  gameRules: GameRules;
  predict?: Predictor;
  planCount?: number;
  randomize?: boolean;
  randomTurnsCount?: number;
}

const defaultPredictor = (gameRules: GameRules) => {
  return async (history: number[]) => {
    let state = gameRules.init();
    let done = false;
    let rewards = [] as number[];
    for (let action of history) {
      const result = gameRules.step(state, action);
      ({ state, done } = result);
    }
    const playerIndex = state.playerIndex;
    const policy = [] as number[];
    const availables = gameRules.availables(state);
    for (let i = 0; i < gameRules.actionsCount; i++) {
      const prob = availables.includes(i + 1) ?
        1 / availables.length : 0;
      policy.push(prob);
    }
    while (!done) {
      const availables = gameRules.availables(state);
      const action = randomOf(availables);
      const result = gameRules.step(state, action);
      ({ state, done, rewards } = result);
    }
    const reward = rewards[playerIndex];
    return {
      policy,
      reward
    };
  };
};

interface LightState {
  readonly playerIndex: number;
  board: number[][] | null;
};

interface LightResult {
  readonly state: LightState;
  readonly rewards: number[];
  readonly done: boolean;
};

interface NodeOptions {
  parent: Node | null;
  action: number;
  prob: number;
  stepResult: LightResult;
}

class Node {
  visits: number;
  parent: Node | null;
  action: number;
  prob: number;
  stepResult: LightResult;
  children: Node[] = [];
  totalValue = 0;
  meanValue = 0;
  prediction?: {
    reward: number,
    policy: number[]
  }
  constructor(options: NodeOptions) {
    this.visits = 0;
    this.parent = options.parent;
    this.action = options.action;
    this.prob = options.prob;
    this.stepResult = options.stepResult;
  }
  isLeaf() {
    return !this.children.length;
  }
  getBonus() {
    const parentVisits = this.parent ?
      this.parent.visits : this.visits;
    const prob = this.prob;
    const bonus = cPuct * prob * Math.sqrt(
      parentVisits
    ) / ( this.visits + 1 );
    return bonus;
  }
  sameAsParent() {
    if (!this.parent) {
      return false;
    } else {
      const parentPlayerIndex =
        this.parent.stepResult.state.playerIndex;
      const playerIndex = this.stepResult.state.playerIndex;
      return parentPlayerIndex === playerIndex;
    }
  }
  findBestLeaf(): Node {
    if (this.isLeaf()) {
      return this;
    }
    const values = this.children.map(node => {
      const sign = node.sameAsParent() ? 1 : -1;
      const value = sign * node.meanValue +
        node.getBonus();
      return value;
    });
    const bestChild = this.children[indexMax(values)];
    return bestChild.findBestLeaf();
  }
  getHistory(): number[] {
    if (!this.parent) {
      return [];
    }
    return this.parent.getHistory().concat(this.action);
  }
  propagate(value: number) {
    this.visits += 1;
    this.totalValue += value;
    this.meanValue = this.totalValue / this.visits;
    if (this.parent) {
      const sign = this.sameAsParent() ? 1 : -1;
      this.parent.propagate(value * sign);
    }
  }
};

export default class Mcts implements PolicyAgent{
  private gameRules: GameRules;
  private predict: Predictor;
  private planCount: number;
  private randomize: boolean;
  private randomTurnsCount?: number;
  private root: Node;
  private rootHistory: number[];
  constructor(options: MctsOptions) {
    this.gameRules = options.gameRules;
    this.predict = options.predict ?
      options.predict : defaultPredictor(this.gameRules);
    this.planCount = options.planCount ?
      options.planCount : options.gameRules.actionsCount;
    this.randomize = !!options.randomize;
    this.randomTurnsCount = options.randomTurnsCount;
    
    this.root = new Node({
      parent: null,
      action: 0,
      prob: 1,
      stepResult: {
        done: false,
        rewards: [],
        state: this.gameRules.init()
      }
    });
    this.rootHistory = [];
  }
  async act() {
    const { action } = await this.policyAct();
    return action;
  }
  async policyAct() {
    await this.plan();
    const policy = Array<number>(this.gameRules.actionsCount)
      .fill(0);
    const probs = this.root.visits > 1 ?
      this.root.children.map(
        child => child.visits / (this.root.visits - 1)
      ) :
      this.root.children.map(
        child => child.prob
      )
    const temp = (
      this.randomTurnsCount &&
      this.rootHistory.length >= this.randomTurnsCount
    ) ? finishTemp : undefined;

    const index = this.randomize ?
      indexSoftMax(probs, temp) :
      indexMax(probs);
    const action = this.root.children[index].action;
    for (let i = 0; i < this.root.children.length; i++) {
      const action = this.root.children[i].action;
      const prob = probs[i];
      policy[action - 1] = prob;
    }
    const value = this.root.meanValue;
    this.step(action);
    return {
      action,
      policy,
      value
    };
  }
  step(action: number) {
    if (printPolicy && this.root.prediction) {
      console.log(this.root.prediction);
    }
    const child = this.root.children.find(
      node => node.action === action
    );
    if (child) {
      this.root = child;
      this.root.parent = null;
    } else {
      const stepResult = this.gameRules.step(
        this.root.stepResult.state as GameState,
        action
      );
      this.root = new Node({
        parent: null,
        action,
        prob: 1,
        stepResult
      });
    }
    this.rootHistory.push(action);
  }
  init() {
    this.root = new Node({
      parent: null,
      action: 0,
      prob: 1,
      stepResult: {
        done: false,
        rewards: [],
        state: this.gameRules.init()
      }
    });
    this.rootHistory = [];
  }
  async expand(node: Node) {
    const history = this.rootHistory.concat(
      node.getHistory()
    );
    const { reward, policy } = await this.predict(history);
    node.prediction = { reward, policy };
    const availables = this.gameRules.availables(
      node.stepResult.state as GameState
    );
    const totalProb = availables.reduce(
      (total, current) => total + policy[current - 1],
      0
    );
    for (let action of availables) {
      const prob = policy[action - 1] / totalProb;
      const stepResult = this.gameRules.step(
        node.stepResult.state as GameState,
        action
      );
      const child = new Node({
        action,
        prob,
        stepResult,
        parent: node
      });
      node.children.push(child);
    }
    node.stepResult.state.board = null;
    return reward;
  }
  async plan() {
    for (let i = 0; i < this.planCount; i++) {
      const node = this.root.findBestLeaf();
      let value = 0;
      if (node.stepResult.done) {
        const reward = node.stepResult.rewards[
          node.stepResult.state.playerIndex
        ];
        value = reward;
      } else {
        value = await this.expand(node);
      }
      node.propagate(value);
    }
  }
}