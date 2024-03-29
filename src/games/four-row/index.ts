import Game from '../../interfaces/game';
import Rules from './rules';
import Model from './model';
import Component from './component';
import GamePlayer, { GamePlayerType } from '../../interfaces/game-player';

export default class FourRow {
    readonly name = 'four-row';
    readonly title = 'Four In A Row';
    readonly rules = new Rules();
    readonly Component = Component;
    readonly players = [
        {type: GamePlayerType.Random},
        {
            type: GamePlayerType.Mcts,
            planCount: 5000
        },
        {
            type: GamePlayerType.Alpha,
            planCount: 300,
            modelName: 'alpha-4'
        },
    ] as GamePlayer[];
    createModel(parallel = false) {
        return new Model(this.name, this.rules, parallel);
    }
}