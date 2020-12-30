import GameRules, {
  Input,
  Output,
  Pair
} from '../../interfaces/game-rules';
import {
  Tile,
  Board,
  State
} from './board';
import { getWinner2D } from '../../lib/xos';
import { getStates } from '../../lib/play';
import HistoryAction from '../../interfaces/history-action';
import { PlaneSymmetry, plane } from '../../lib/transforms';
import { oneHot } from '../../lib/helpers';
import GameHistory from '../../interfaces/game-history';

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
  positionToAction(position: { i: number, j: number }) {
    const { i, j } = position;
    const action = i * this.width + j + 1;
    return action;
  }
  getInput(history: number[]) {
    const states = getStates(history, this);
    const lastState = states[states.length - 1];
    const playerIndex = lastState.playerIndex;
    const enemyIndex = 1 - playerIndex;
    const playerTile = playerIndex === 0 ? Tile.X : Tile.O;
    const enemyTile = playerIndex === 1 ? Tile.X : Tile.O;

    const input = [] as Input;

    for (let i = 0; i < this.height; i++) {
      input.push([]);
      for (let j = 0; j < this.width; j++) {
        input[i].push([]);
        const color = 1 - playerIndex;
        const playerHistory = states
          .filter(state => state.playerIndex === enemyIndex)
          .map(state => (
              state.board[i][j] === playerTile ? 1 : 0
          ))
          .reverse()
          .slice(0, historyDepth) as number[];
        const enemyHistory = states
          .filter(state => state.playerIndex === playerIndex)
          .map(state => (
              state.board[i][j] === enemyTile ? 1 : 0
          ))
          .reverse()
          .slice(0, historyDepth) as number[];
        const emptyPlayerHistory = [] as number[];
        const emptyEnemyHistory = [] as number[];
        const emptyPlayerHistoryLength = Math.max(
          historyDepth - playerHistory.length, 0
        );
        const emptyEnemyHistoryLength = Math.max(
          historyDepth - enemyHistory.length, 0
        );
        if (emptyPlayerHistoryLength) {
          for (let n = 0; n < emptyPlayerHistoryLength; n++) {
            emptyPlayerHistory.push(0);
          }
        }
        if (emptyEnemyHistoryLength) {
          for (let n = 0; n < emptyEnemyHistoryLength; n++) {
            emptyEnemyHistory.push(0);
          }
        }
        input[i][j] = playerHistory.concat(
          emptyPlayerHistory,
          enemyHistory,
          emptyEnemyHistory,
          useColor ? [color] : []
        );
      }
    }
    return input;
  }
  private getSymHistories(history: HistoryAction[]) {
    const symHistories = [history];
    const syms = [
      PlaneSymmetry.Horizontal,
      PlaneSymmetry.Vertical,
      PlaneSymmetry.Rotation180
    ].concat(this.width === this.height ? 
      [
        PlaneSymmetry.DiagonalSlash,
        PlaneSymmetry.DiagonalBackSlash,
        PlaneSymmetry.Rotation90,
        PlaneSymmetry.Rotation270
      ] : []
    );

    for (let sym of syms) {
      const symHistory = history.map(({ action, best, value }) => {
        const actionPosition = this.actionToIJ(action);
        const symPosition = plane({
          i: actionPosition.i,
          j: actionPosition.j,
          height: this.height,
          width: this.width,
          sym
        });
        const symAction = this.positionToAction(symPosition);
        const bestPosition = this.actionToIJ(best);
        const symBestPosition = plane({
          i: bestPosition.i,
          j: bestPosition.j,
          height: this.height,
          width: this.width,
          sym
        });

        const symBest = this.positionToAction(symBestPosition);

        return {
          action: symAction,
          best: symBest,
          value
        } as HistoryAction
      });
      symHistories.push(symHistory);
    }
    return symHistories;
  }
  private getOutput(gameHistory: GameHistory) {
    const lastHistoryAction = gameHistory.history[
      gameHistory.history.length - 1
    ];
    const policyOutput = oneHot(lastHistoryAction.best, this.actionsCount);
    const rewardOutput = lastHistoryAction.value;
    return [policyOutput, rewardOutput] as Output;
  }
  getPairs(gameHistory: GameHistory) {
    const pairs = [] as Pair[];
    const symHistories = this.getSymHistories(gameHistory.history);
    for (let historyActions of symHistories) {
      for (let i = 1; i <= historyActions.length; i++) {
        const subHistoryActions = historyActions.slice(0, i);
        const subHistory = subHistoryActions.map(({action}) => action);
        const input = this.getInput(subHistory);
        const output = this.getOutput({
          history: subHistoryActions,
          rewards: gameHistory.rewards
        });
        pairs.push({
          input,
          output
        });
      }
    }
    return pairs;
  }
}