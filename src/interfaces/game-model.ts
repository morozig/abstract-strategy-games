import GameHistory from './game-history';
import GamePrediction from './game-prediction';

export interface TrainOptions {
    improve?: boolean;
}

export default interface GameModel {
    train (
        gameHistories: GameHistory[],
        options?: TrainOptions
    ): Promise<boolean>;
    save (name: string): Promise<void>;
    load (name: string): Promise<void>;
    predict(history: number[]): Promise<GamePrediction>;
}