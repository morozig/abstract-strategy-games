import GameHistory from './game-history';
import GameState from './game-state';
import GameStepResult from './game-step-result';

export type Input = number[][][];
export type Output = [number[], number];

export interface Pair {
  input: Input;
  output: Output;
};

export default interface Rules{
  step (gameState: GameState, action: number): GameStepResult;
  availables(gameState: GameState): number[];
  init() : GameState;
  actionsCount: number;
  height: number;
  width: number;
  depth: number;
  playersCount: number;
  getPairs(gameHistory: GameHistory): Pair[];
  getInput(history: number[]): Input;
}