import React from 'react';
import './Tile.css';
import TileNumber from '../tile';

interface TileProps {
    tile: TileNumber;
    x: number;
    y: number;
    onClick: () => void;
}

const Tile: React.FC<TileProps> = (props) => {
    return (
        <svg
            viewBox={'0 0 10 10'}
            x={props.x}
            y={props.y}
            width={10}
            height={10}
        >
            <circle
                cx={5}
                cy={5}
                r={3.5}
                className={`Tile Tile--${TileNumber[props.tile]}`}
                onClick={(e) => {
                    e.preventDefault();
                    props.onClick();
                }}
            />
        </svg>
    );
};

export default Tile;
