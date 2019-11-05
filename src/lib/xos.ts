const getWinner1D = (row: number[], sameCount: number) => {
    if (row.length < sameCount) {
        return 0;
    }
    const winner = row.find(
        (value, index) => {
            if (!value) {
                return false
            }
            const nextValues = row.slice(index, index + sameCount);
            return nextValues.length === sameCount &&
                nextValues
                    .every(other => other === value);
        }
    );
    return winner || 0;
}

interface Point {
    i: number;
    j: number;
}

const slice2D = (board: number[][], start: Point, step: Point) => {
    const row = [];
    let position = {...start};
    while (true) {
        if (board[position.i] === undefined) break;
        const value = board[position.i][position.j];
        if (value === undefined) break;
        row.push(value);
        position = {
            i: position.i + step.i,
            j: position.j + step.j
        }
    }
    return row;
};

const getWinner2D = (board: number[][], sameCount: number) => {
    const rows = board;
    for (let row of rows) {
        const winner = getWinner1D(row, sameCount);
        if (winner) return winner;
    }

    const width = board[0].length;
    const heigth = board.length;

    const columns = Array.from(Array(width).keys())
        .map(j => slice2D(board, {i: 0, j}, {i: 1, j: 0}));
    for (let column of columns) {
        const winner = getWinner1D(column, sameCount);
        if (winner) return winner;
    }

    const rightDiagsTop = Array.from(Array(width).keys())
        .map(j => slice2D(board, {i: 0, j}, {i: 1, j: 1}));
    for (let diag of rightDiagsTop) {
        const winner = getWinner1D(diag, sameCount);
        if (winner) return winner;
    }

    const rightDiagsBottom = Array.from(Array(heigth).keys())
        .slice(1)
        .map(i => slice2D(board, {i, j: 0}, {i: 1, j: 1}));
    for (let diag of rightDiagsBottom) {
        const winner = getWinner1D(diag, sameCount);
        if (winner) return winner;
    }

    const leftDiagsTop = Array.from(Array(width).keys())
        .map(j => slice2D(board, {i: 0, j}, {i: 1, j: -1}));
    for (let diag of leftDiagsTop) {
        const winner = getWinner1D(diag, sameCount);
        if (winner) return winner;
    }

    const leftDiagsBottom = Array.from(Array(heigth).keys())
        .slice(1)
        .map(i => slice2D(board, {i, j: width - 1}, {i: 1, j: -1}));
    for (let diag of leftDiagsBottom) {
        const winner = getWinner1D(diag, sameCount);
        if (winner) return winner;
    }

    return 0;
};

export {
    getWinner1D,
    getWinner2D
};