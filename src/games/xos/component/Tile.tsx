import React from 'react';
import './Tile.css';
import { Tile as TileNumber } from '../board';

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
            className={`Tile`}
        >
            <rect
                x={0}
                y={0}
                width={10}
                height={10}
                className={'Tile-border'}
            />
            {props.tile === TileNumber.X && 
                <g
                    className={'Tile--X'}
                >
                    <line
                        x1={1}
                        y1={1}
                        x2={9}
                        y2={9}
                    />
                    <line
                        x1={1}
                        y1={9}
                        x2={9}
                        y2={1}
                    />
                </g>
            }
            {props.tile === TileNumber.O && 
                <circle
                    className={'Tile--O'}
                    cx={5}
                    cy={5}
                    r={4}
                />
            }
            {props.tile === TileNumber.Empty && 
                <rect
                    x={0}
                    y={0}
                    width={10}
                    height={10}
                    className={'Tile--Empty'}
                    onClick={(e) => {
                        e.preventDefault();
                        props.onClick();
                    }}
                />
            }
        </svg>
    );
};

export default Tile;
