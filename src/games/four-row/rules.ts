import GameRules from '../../interfaces/game-rules';
import Action from './action';
import Tile from './tile';
import Board from './board';
import State from './state';
import { getWinner2D } from '../../lib/xos';

const actionToIJ = (board: Board, action: Action) => {
    const j = action - 1;
    let i = 5;
    while (board[i][j] !== 0 && i >= 0) {
        i--;
    }
    return [i, j];
};

const copy = (board: Board) => board.map(row => row.slice()) as Board;

export default class Rules {
    step(state: State, action: Action) {
        if (!this.availables(state).includes(action)){
            throw new Error('bad action!');
        }
        const board = copy(state.board);
        const playerTile = state.playerIndex + 1;
        const [ actionI, actionJ ] = actionToIJ(board, action);
        board[actionI][actionJ] = playerTile;
        const winner = getWinner2D(board, 4);
        const done = (!!winner ||
            this.availables({
                board,
                playerIndex: -1
            }).length === 0
        );
        const rewards = [0, 0];
        if (winner) {
            rewards[state.playerIndex] = 1;
            rewards[1 - state.playerIndex] = -1;
        }
        const newPlayerIndex = done ?
            state.playerIndex : 1 - state.playerIndex;
        const newState = {
            board,
            playerIndex: newPlayerIndex
        };
        const gameStepResult = {
            done,
            rewards,
            state: newState
        };
        return gameStepResult;
    }
    availables(state: State) {
        const actions = [] as Action[];
        const board = state.board;
        for (let j = 0; j < 7; j++) {
            if (board[0][j] === Tile.Empty) {
                actions.push(j + 1);
            }
        }
        return actions;
    }
    init() {
        const board = [
            [Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty],
            [Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty],
            [Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty],
            [Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty],
            [Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty],
            [Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty, Tile.Empty]
        ] as Board;
        const playerIndex = 0;
        return {board, playerIndex};
    }
    actionsCount = 7
}