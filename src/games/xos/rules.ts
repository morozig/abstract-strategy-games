import GameRules from '../../interfaces/game-rules';
import {
  Tile,
  Board,
  State
} from './board';
import { getWinner2D } from '../../lib/xos';
import { getStates } from '../../lib/play';

const historyDepth = 2;
const useColor = true;

const copy = (board: Board) => board.map(row => row.slice()) as Board;

export default class Rules implements GameRules {
  readonly height: number;
  readonly width: number;
  readonly actionsCount: number;
  readonly depth: number;
  private same: number;
  constructor(height: number, width: number, same: number) {
    this.height = height;
    this.width = width;
    this.same = same;
    this.actionsCount = height * width;
    this.depth = 2 * historyDepth + (useColor ? 1 : 0);
  }
  step(state: State, action: number) {
    if (!this.availables(state).includes(action)){
      throw new Error('bad action!');
    }
    const board = copy(state.board);
    const playerTile = state.playerIndex + 1;
    const { i , j } = this.actionToIJ(action);
    board[i][j] = playerTile;
    const winner = getWinner2D(board, this.same);
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
    const actions = [] as number[];
    const board = state.board;
    for (let i = 0; i < this.height; i++) {
      for (let j = 0; j < this.width; j++) {
        if (board[i][j] === Tile.Empty) {
          actions.push(i * this.width + j + 1);
        }
      }
    }
    return actions;
  }
  init() {
    const board = [] as Board;
    for (let i = 0; i < this.height; i++) {
      board[i] = [];
      for (let j = 0; j < this.width; j++) {
        board[i][j] = Tile.Empty;
      }
    }
    const playerIndex = 0;
    return {board, playerIndex};
  }
  private actionToIJ(action: number) {
    const i = Math.floor((action - 1) / this.width);
    const j = (action - 1) % this.width;
    return { i, j };
  }
  private positionToAction(position: { i: number, j: number }) {
    const { i, j } = position;
    const action = i * this.width + j + 1;
    return action;
  }
  getInput(history: number[]) {
    const states = getStates(history, this);
  }
}