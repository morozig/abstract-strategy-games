import Game from '../../interfaces/game';
import Rules from './rules';
import { component } from './component';
import GamePlayer, { GamePlayerType } from '../../interfaces/game-player';
import GameComponent from '../../interfaces/game-component';
import AlphaModel from '../../lib/alpha-model';
import Policy from './networks/policy';
import Reward from './networks/reward';
import {
  Worker
} from 'threads';
import config from './config.json';

export default class Xos implements Game {
  readonly height = config.height;
  readonly width = config.width;
  readonly same = config.same;
  readonly name: string;
  readonly title: string;
  readonly rules: Rules;
  readonly Component: GameComponent;
  readonly players = [
    {type: GamePlayerType.Random},
    {
      type: GamePlayerType.Mcts,
      planCount: 5000
    },
    {
      type: GamePlayerType.Alpha,
      planCount: config.planCount,
      modelName: 'mcts-5000'
    },
  ] as GamePlayer[];
  constructor() {
    this.name = `xos${this.height}${this.width}${this.same}`;
    this.title = `${this.height},${this.width},${this.same}-game`;
    this.rules = new Rules(this.height, this.width, this.same);
    this.Component = component(this.rules);
  }
  createWorker() {
    return new Worker('./worker', {
      name: 'xos'
    });
  }
  createModel() {
    const policy = new Policy({
      height: this.rules.height,
      width: this.rules.width,
      depth: this.rules.depth
    });
    const reward = new Reward({
      height: this.rules.height,
      width: this.rules.width,
      depth: this.rules.depth
    });
    return new AlphaModel({
      gameName: this.name,
      rules: this.rules,
      policy,
      reward
    });
  }
}