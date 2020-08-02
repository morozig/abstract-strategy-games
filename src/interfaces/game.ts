import GameRules from './game-rules';
import GameModel from './game-model';
import GameComponent from './game-component';
import GamePlayer from './game-player';

export default interface Game {
    readonly name: string;
    readonly title: string;
    readonly rules: GameRules;
    readonly Component: GameComponent;
    readonly players: GamePlayer[];
    readonly randomTurnsCount: number;
    createModel(parallel?: boolean): GameModel;
};