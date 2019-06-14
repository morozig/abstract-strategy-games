import * as React from 'react';

interface TileProps {
    tile: number;
    onClick: () => void;
}

class Tile extends React.Component<TileProps> {
    render() {
        const tile = this.props.tile;
        let fill;
        switch (tile){
            case 0: {
                fill = 'white'
                break;
            }
            case 1: {
                fill = 'yellow'
                break;
            }
            case 2: {
                fill = 'red'
                break;
            }
        }
        return <circle 
            cx="30" cy="30" r="25" 
            fill={fill}
            onClick = {this.props.onClick}
        />;
    }
}

export default Tile;
