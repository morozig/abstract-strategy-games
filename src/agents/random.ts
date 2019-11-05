import Agent from '../interfaces/agent';
import GameRules from '../interfaces/game-rules';
import GameState from '../interfaces/game-state';
import { randomOf } from '../lib/helpers';

export default class Random implements Agent{
    private gameRules: GameRules;
    private gameState: GameState
    constructor(gameRules: GameRules) {
        this.gameRules = gameRules;
        this.gameState = gameRules.init();
    }
    async act() {
        const availables = this.gameRules.availables(this.gameState);
        const action = randomOf(availables);
        this.step(action);
        return action;
    }
    step(action: number) {
        const gameStepResult = this.gameRules.step(this.gameState, action);
        this.gameState = gameStepResult.state;
    }
    init() {
        this.gameState = this.gameRules.init();
    }
}