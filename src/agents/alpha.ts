import Agent from '../interfaces/agent';
import GameRules from '../interfaces/game-rules';
import GameModel from '../interfaces/game-model';
import Mcts from './mcts';

interface AlphaOptions {
    gameRules: GameRules;
    model: GameModel;
    url?: string;
    planCount?: number;
    randomize?: boolean;
}

const modelPredictor = (model: GameModel, url?: string) => {
    let modelLoaded = !url;
    return async(history: number[]) => {
        if (!modelLoaded && url) {
            await model.load(url);
            modelLoaded = true;
        }
        return await model.predict(history);
    };
};

export default class Alpha implements Agent {
    private mcts: Mcts;
    constructor(options: AlphaOptions) {
        this.mcts = new Mcts({
            gameRules: options.gameRules,
            predict: modelPredictor(options.model, options.url),
            planCount: options.planCount,
            randomize: options.randomize
        });
    }
    act() {
        return this.mcts.act();
    }
    init() {
        return this.mcts.init();
    }
    step(action: number) {
        return this.mcts.step(action);
    }
};