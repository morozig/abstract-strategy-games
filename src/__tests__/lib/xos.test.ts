import {
    getWinner1D,
    getWinner2D
} from '../../lib/xos'

test('row 1', () => {
    const row = [0, 0, 0, 0, 0, 0, 0];
    const expected = 0;
    const actual = getWinner1D(row, 4);
    expect(actual).toBe(expected);
});

test('row 2', () => {
    const row = [1, 1, 1, 1, 0, 0, 0];
    const expected = 1;
    const actual = getWinner1D(row, 4);
    expect(actual).toBe(expected);
});

test('row 3', () => {
    const row = [2, 2, 2, 2, 0, 0, 0];
    const expected = 2;
    const actual = getWinner1D(row, 4);
    expect(actual).toBe(expected);
});

test('row 4', () => {
    const row = [0, 0, 0, 1, 1, 1, 1];
    const expected = 1;
    const actual = getWinner1D(row, 4);
    expect(actual).toBe(expected);
});

test('row 5', () => {
    const row = [0, 0, 0, 1, 1, 1, 1];
    const expected = 0;
    const actual = getWinner1D(row, 5);
    expect(actual).toBe(expected);
});

test('row 6', () => {
    const row = [0, 0, 2, 2, 2, 2, 0];
    const expected = 2;
    const actual = getWinner1D(row, 4);
    expect(actual).toBe(expected);
});

test('row 7', () => {
    const row = [1, 1, 1, 2, 2, 2, 2];
    const expected = 2;
    const actual = getWinner1D(row, 4);
    expect(actual).toBe(expected);
});


test('board 1', () => {
    const board = [
        [1, 2, 1],
        [2, 1, 2],
        [2, 1, 2],
    ];
    const expected = 0;
    const actual = getWinner2D(board, 3);
    expect(actual).toBe(expected);
});

test('board 2', () => {
    const board = [
        [1, 2, 1],
        [2, 1, 2],
        [1, 1, 2],
    ];
    const expected = 1;
    const actual = getWinner2D(board, 3);
    expect(actual).toBe(expected);
});

test('board 3', () => {
    const board = [
        [2, 2, 1],
        [2, 2, 1],
        [1, 1, 2],
    ];
    const expected = 2;
    const actual = getWinner2D(board, 3);
    expect(actual).toBe(expected);
});

test('board 4', () => {
    const board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [2, 2, 2, 1, 0, 0, 0]
    ];
    const expected = 1;
    const actual = getWinner2D(board, 4);
    expect(actual).toBe(expected);
});

test('board 5', () => {
    const board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0],
        [2, 2, 1, 0, 0, 0, 0],
        [2, 2, 2, 1, 0, 0, 0]
    ];
    const expected = 1;
    const actual = getWinner2D(board, 4);
    expect(actual).toBe(expected);
});

test('board 6', () => {
    const board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 1],
        [2, 0, 0, 2, 2, 0, 2],
        [2, 0, 0, 2, 1, 1, 1],
        [1, 2, 2, 2, 1, 1, 1]
    ];
    const expected = 2;
    const actual = getWinner2D(board, 4);
    expect(actual).toBe(expected);
});

test('board 7', () => {
    const board = [
        [2, 2, 2, 1, 2, 2, 2],
        [1, 1, 1, 2, 1, 1, 1],
        [2, 2, 2, 1, 2, 2, 2],
        [1, 1, 1, 2, 1, 1, 1],
        [2, 2, 2, 1, 2, 2, 2],
        [1, 1, 1, 2, 1, 1, 1]
    ];
    const expected = 0;
    const actual = getWinner2D(board, 4);
    expect(actual).toBe(expected);
});

