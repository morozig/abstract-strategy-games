import React, { useRef } from 'react';
import './App.css';
import GameComponent from '../game';
// import Game from '../../games/four-row';
import Game from '../../games/xos';

const App: React.FC = () => {
    const gameRef = useRef(new Game(5, 5, 4));

    return (
        <div className={'App'}>
            <GameComponent
                game={gameRef.current}
            />
        </div>
    );
}

export default App;
