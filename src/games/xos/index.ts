import Game from '../../interfaces/game';
import Rules from './rules';
import Model from './model';
import { component } from './component';
import GamePlayer, { GamePlayerType } from '../../interfaces/game-player';
import GameComponent from '../../interfaces/game-component';

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
            planCount: 500
        },
        {
            type: GamePlayerType.Alpha,
            planCount: 50,
            modelName: 'alpha-12'
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
    createModel(parallel = false) {
        return new Model(this.name, this.rules, parallel);
    }
}