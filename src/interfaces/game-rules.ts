import GameState from './game-state';
import GameStepResult from './game-step-result';

export default interface Rules{
    step (gameState: GameState, action: number): GameStepResult;
    availables(gameState: GameState): number[];
    init() : GameState;
    actionsCount: number;
}