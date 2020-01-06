import React from 'react';
import State from '../state';
import './Component.css';
import Tile from './Tile';
import GameComponent from '../../../interfaces/game-component';

const Component: GameComponent = (props) => {
    const state = props.gameState as State;

    return (
        <svg
            className={'Component'}
            viewBox={'0 0 70 60'}
        >
            <rect
                x={0}
                y={0}
                width={70}
                height={60}
                rx={2}
                ry={2}
                className={'Background'}
            />
            {state.board.map((row, i) => row.map((tile, j) => (
                <Tile
                    key={i * 6 + j}
                    tile={tile}
                    x={j * 10}
                    y={i * 10}
                    onClick={() => {
                        const action = j + 1;
                        props.onAction(action);
                    }}
                />
            )))}
        </svg>
    );
};

export default Component;
