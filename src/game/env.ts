class State {
    player: number;
    board: number[][];
    constructor(initState?: State){
        this.player = 1;
        this.board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ] as number[][];
        if (initState) {
            for (let i in initState.board) {
                for (let j in initState.board[i]) {
                    this.board[i][j] = initState.board[i][j];
                }
            }
            this.player = initState.player;
        }
    }
}

abstract class Env {
    static step(state: State, action: number) {
        if (!this.availables(state).includes(action)){
            throw ('bad action!');
        }
        const player = state.player;
        const newState = new State(state);
        const [ actionI, actionJ ] = actionToIJ(state, action);
        newState.board[actionI][actionJ] = player;
        const winner = hasWinner(newState);
        let done = false;
        let reward = 0;
        if (winner) {
            done = true;
            reward = winner === player ? 1 : -1;
        } else {
            done = this.availables(newState).length === 0;
        }
        if (!done) {
            newState.player = 3 - player;
        }
        return [newState, reward, done];
    }
    static availables(state: State) {
        const actions = [] as number[];
        const board = state.board;
        for (let j = 0; j < 7; j++) {
            if (board[0][j] === 0) {
                actions.push(j + 1);
            }
        }
        return actions;
    }
    static init() {
        return new State();
    }
}

const actionToIJ = (state: State, action: number) => {
    const j = action - 1;
    const board = state.board;
    let i = 5;
    while (board[i][j] !== 0 && i >= 0) {
        i--;
    }
    return [i, j];
};

// const iJtoAction = (i: number, j: number) => {
//     const index = i * 3 + j;
//     const action = index + 1;
//     return action;
// };

const hasWinner = (state: State) => {
    const board = state.board;
    for (let row of board) {
        let winner = fourInARow(row);
        if (winner) {
            return winner;
        }
    }

    for (let column of Array(7).fill(0).map(
        (_, j) => board.map(
            (_, i) => board[i][j]
        )
    )){
        let winner = fourInARow(column);
        if (winner) {
            return winner;
        }
    }

    const diags = [] as any[];
    for (let [i, j] of [[2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [0, 3]]) {
        diags.push(getDiag(board, i, j, 1));
    }
    for (let [i, j] of [[2, 6], [1, 6], [0, 6], [0, 5], [0, 4], [0, 3]]) {
        diags.push(getDiag(board, i, j, -1));
    }
    for (let diag of diags) {
        let winner = fourInARow(diag);
        if (winner) {
            return winner;
        }
    }
    return 0;
};

const allSame = (tiles: number[]) => {
    return tiles.every(tile => tile === tiles[0]);
}

const fourInARow = (tiles: number[]) => {
    for (let i = 0; i <= tiles.length - 4; i++){
        if (tiles[i]) {
            if (allSame(tiles.slice(i, i + 4))){
                return tiles[i];
            }
        }
    }
    return false;
}

const getDiag = (board: number[][], startI, startJ, sign) => {
    const tiles = [] as any[];
    if (sign > 0) {
        for (let i = startI, j = startJ; i < 6 && j < 7; i++, j++) {
            tiles.push(board[i][j]);
        }
    } else {
        for (let i = startI, j = startJ; i < 6 && j >= 0; i++, j--) {
            tiles.push(board[i][j]);
        }
    }
    return tiles;
};

// const rotate = (board: number[][], probs: number[]) => {
//     return [
//         [
//             [board[2][0], board[1][0], board[0][0]],
//             [board[2][1], board[1][1], board[0][1]],
//             [board[2][2], board[1][2], board[0][2]]
//         ],
//         [
//             probs[6], probs[3], probs[0], probs[7], probs[4], 
//             probs[1], probs[8], probs[5], probs[2]
//         ]
//     ];
// };

// const verticalSym = (board: number[][], probs: number[]) => {
//     return [
//         [
//             [board[2][0], board[2][1], board[2][2]],
//             [board[1][0], board[1][1], board[1][2]],
//             [board[0][0], board[0][1], board[0][2]]
//         ],
//         [
//             probs[6], probs[7], probs[8], probs[3], probs[4], 
//             probs[5], probs[0], probs[1], probs[2]
//         ]
//     ];
// };

const horizontalSym = (board: number[][], probs: number[]) => {
    return [
        [
            [
                board[0][6], board[0][5], board[0][4], board[0][3],
                board[0][2], board[0][1], board[0][0]
            ],
            [
                board[1][6], board[1][5], board[1][4], board[1][3],
                board[1][2], board[1][1], board[1][0]
            ],
            [
                board[2][6], board[2][5], board[2][4], board[2][3],
                board[2][2], board[2][1], board[2][0]
            ],
            [
                board[3][6], board[3][5], board[3][4], board[3][3],
                board[3][2], board[3][1], board[3][0]
            ],
            [
                board[4][6], board[4][5], board[4][4], board[4][3],
                board[4][2], board[4][1], board[4][0]
            ],
            [
                board[5][6], board[5][5], board[5][4], board[5][3],
                board[5][2], board[5][1], board[5][0]
            ]
        ],
        [
            probs[6], probs[5], probs[4], probs[3],
            probs[2], probs[1], probs[0]
        ]
    ];
};

const getEquiData = (board: number[][], probs: number[]) => {
    const result = [
        [board, probs]
    ];
    result.push(horizontalSym(board, probs));
    return result;
};

export {
    Env,
    getEquiData
}