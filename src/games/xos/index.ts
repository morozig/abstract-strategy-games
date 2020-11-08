import Game from '../../interfaces/game';
import Rules from './rules';
import { component } from './component';
import GamePlayer, { GamePlayerType } from '../../interfaces/game-player';
import GameComponent from '../../interfaces/game-component';
import AlphaModel from '../../lib/alpha-model';
import * as network from './networks/residual';

export default class Xos implements Game {
  readonly height: number;
  readonly width: number;
  readonly same: number;
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
      planCount: 300,
      modelName: 'mcts-5000'
    },
  ] as GamePlayer[];
  constructor(height: number, width: number, same: number) {
    this.height = height;
    this.width = width;
    this.same = same;
    this.name = `xos${this.height}${this.width}${this.same}`;
    this.title = `${this.height},${this.width},${this.same}-game`;
    this.rules = new Rules(this.height, this.width, this.same);
    this.Component = component(this.rules);
  }
  createModel() {
    const {
      createModel,
      ...modelParams
    } = network;
    const model = createModel({
      height: this.rules.height,
      width: this.rules.width,
      depth: this.rules.depth
    });
    return new AlphaModel({
      gameName: this.name,
      rules: this.rules,
      model,
      ...modelParams
    });
  }
}