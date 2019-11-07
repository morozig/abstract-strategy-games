enum GamePlayerType {
    Human,
    Alpha,
    Random,
    Mcts
}

export default interface GamePlayer {
    type: GamePlayerType;
    name?: string;
    modelName?: string;
    planCount?: number;
};

export {
    GamePlayerType
}