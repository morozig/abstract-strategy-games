import * as React from 'react';
import Tile from './Tile';
import './Board.css';

interface BoardProps {
    board: number[][];
    onAction: (action: number) => void;
}

const iJtoAction = (i: number, j: number) => {
    const action = j + 1;
    return action;
};

const tiles = (board: number[][], onAction: (action: number) => void) => {
    const rows = board.map((row, i ) => 
        <svg className='Row' x = "0" y = {'' + i * 60}
            key = {i.toString()}>
            {row.map((tile, j) => 
                <svg x = {'' + j * 60} y = "0" key = {j.toString()}>
                    <Tile tile = {tile}
                    onClick = {() => {
                        const action = iJtoAction(i, j);
                        onAction(action);
                    }}
                    />
                </svg>
            )}
        </svg>
    );
    return rows;
};

class Board extends React.Component<BoardProps> {
    render() {
        return (
            <div className="Board">
                <svg width="420" height="360">
                    <svg className="BackGround">
                        <rect x="0" y="0" width="420"
                            height="360" rx="15" ry="15"
                        />
                        {tiles(this.props.board, this.props.onAction)}
                    </svg>
                </svg>
            </div>
        );
    }
}

export default Board;
