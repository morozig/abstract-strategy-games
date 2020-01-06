import GameState from '../../interfaces/game-state';

export enum Tile {
    Empty,
    X,
    O
}

export type Board = Tile[][];

export interface State extends GameState {
    readonly board: Tile[][];
}
