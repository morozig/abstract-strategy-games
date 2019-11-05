enum GamePlayerType {
    Human,
    Alpha,
    Random,
    Mcts
}

export default interface GamePlayer {
    type: GamePlayerType;
    name?: string;
    url?: string;
    planCount?: number;
};

export {
    GamePlayerType
}