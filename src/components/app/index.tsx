import React, { useRef } from 'react';
import './App.css';
import Game from '../game';
import FourRow from '../../games/four-row';

const App: React.FC = () => {
    const gameRef = useRef(new FourRow());

    return (
        <div className={'App'}>
            <Game
                game={gameRef.current}
            />
        </div>
    );
}

export default App;
