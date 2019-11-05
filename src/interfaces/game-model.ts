import GameHistory from './game-history';
import GamePrediction from './game-prediction';

export default interface GameModel {
    train (gameHistories: GameHistory[]): Promise<boolean>;
    save (name: string): Promise<void>;
    load (name: string): Promise<void>;
    predict(history: number[]): Promise<GamePrediction>;
}