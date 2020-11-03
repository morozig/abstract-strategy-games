import GameRules from './game-rules';
import GameComponent from './game-component';
import GamePlayer from './game-player';
import AlphaModel from '../lib/alpha-model';

export default interface Game {
    readonly name: string;
    readonly title: string;
    readonly rules: GameRules;
    readonly Component: GameComponent;
    readonly players: GamePlayer[];
    readonly randomTurnsCount: number;
    createModel(): AlphaModel;
};