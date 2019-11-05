import Board from './board';
import GameState from '../../interfaces/game-state';

export default interface State extends GameState {
    readonly board: Board;
}