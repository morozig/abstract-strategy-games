import React, { useState } from 'react';
import './Game.css';
import GameInterface from '../../interfaces/game';
import { useGame } from './hooks';
import GamePlayer, { GamePlayerType } from '../../interfaces/game-player';

interface GameProps {
    game: GameInterface;
}

const Game: React.FC<GameProps> = (props) => {
    const Component = props.game.Component;
    const {
        gameState,
        status,
        onAction,
        start,
        stop
    } = useGame(props.game);
    const isRunning = status.includes('turn');
    const gamePlayers = props.game.players.find(
        player => player.type === GamePlayerType.Human
    ) ? props.game.players : [{
        type: GamePlayerType.Human
    } as GamePlayer].concat(props.game.players);
    const defaultPlayers = [0];
    const playersCount = 2;

    for (let i = 1; i <= playersCount - 1; i++) {
        defaultPlayers.push(gamePlayers.length - 1);
    }

    const [ players, setPlayers ] = useState(defaultPlayers);

    return (
        <div className={'Game'}>
            <h2>
                {props.game.title}
            </h2>
            <Component
                gameState={gameState}
                onAction={onAction}
            />
            <div className={'Game-status'}>
                <p className={'Game-text'}>
                    {status}
                </p>
            </div>
            <div className={'Game-controls'}>
                <div>
                    {players.map((playerIndex, j) => (
                        <div
                            className={'Game-select'}
                            key={j}
                        >
                            <p className={'Game-text'}>
                                {`Player ${j + 1}:`}
                            </p>
                            <select
                                value={playerIndex}
                                onChange={e => {
                                    const newPlayerIndex = +e.target.value;
                                    setPlayers(players => {
                                        const newPlayers = players.slice();
                                        newPlayers.splice(
                                            j, 1, newPlayerIndex
                                        );
                                        return newPlayers;
                                    });
                                }}
                            >
                                {gamePlayers.map((player, i) => (
                                    <option
                                        key={i}
                                        value={i}
                                    >
                                        {
                                            player.name ||
                                            GamePlayerType[player.type]
                                        }
                                    </option>
                                ))}
                            </select>
                        </div>
                    ))}
                </div>
                {isRunning ? 
                    <button
                        className={'Game-button'}
                        onClick={e => {
                            e.preventDefault();
                            stop();
                        }}
                    >
                        <p className={'Game-text'}>
                            {'Stop'}
                        </p>
                    </button> :
                    <button
                        className={'Game-button'}
                        onClick={e => {
                            e.preventDefault();
                            start(players.map(index => gamePlayers[index]));
                        }}
                    >
                        <p className={'Game-text'}>
                            {'Start'}
                        </p>
                    </button>
                }
            </div>
        </div>
    );
}

export default Game;
