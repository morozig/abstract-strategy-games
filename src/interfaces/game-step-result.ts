import GameState from "./game-state";

export default interface StepResult {
  readonly state: GameState;
  readonly rewards: number[];
  readonly done: boolean;
};