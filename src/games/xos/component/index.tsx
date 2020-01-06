import React from 'react';
import { State } from '../board';
import './Component.css';
import Tile from './Tile';
import GameComponent from '../../../interfaces/game-component';
import Rules from '../rules';

const component = (rules: Rules) => {
    const Component: GameComponent = (props) => {
        const state = props.gameState as State;
    
        return (
            <svg
                className={'Component'}
                viewBox={`0 0 ${rules.width * 10} ${rules.height * 10}`}
            >
                <rect
                    x={0}
                    y={0}
                    width={rules.width * 10}
                    height={rules.height * 10}
                    className={'Background'}
                />
                {state.board.map((row, i) => row.map((tile, j) => (
                    <Tile
                        key={i * rules.width + j}
                        tile={tile}
                        x={j * 10}
                        y={i * 10}
                        onClick={() => {
                            const action = rules.positionToAction({i, j});
                            if (rules.availables(state).includes(action)) {
                                props.onAction(action);
                            }
                        }}
                    />
                )))}
            </svg>
        );
    };
    return Component;
};


export {
    component
}
    
