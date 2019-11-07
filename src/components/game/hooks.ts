import { useState, useCallback, useRef } from 'react';
import Game from '../../interfaces/game';
import GamePlayer, { GamePlayerType } from '../../interfaces/game-player';
import Random from '../../agents/random';
import Human from '../../agents/human';
import Agent from '../../interfaces/agent';
import Mcts from '../../agents/mcts';
import { sleep } from '../../lib/helpers';
import Alpha from '../../agents/alpha';

type ActionResolve = ((action: number) => void) | null; 

const useGame = (game: Game) => {
    const [ gameState, setGameState ] = useState(game.rules.init());
    const [ status, setStatus ] = useState('Choose players and start');
    const isRunningRef = useRef(false);
    const actionResolveRef = useRef<ActionResolve>(null);
    const onAction = useCallback((action: number) => {
        if (actionResolveRef.current) {
            actionResolveRef.current(action);
            actionResolveRef.current = null;
        }
    }, []);
    const humanAction = useCallback(() => {
        return new Promise((resolve: ActionResolve) => {
            actionResolveRef.current = resolve
        });
    }, []);
    const start = useCallback((players: GamePlayer[]) => {
        const agents = players.map(player => {
            switch (player.type) {
                case GamePlayerType.Random: {
                    return new Random(game.rules);
                }
                case GamePlayerType.Mcts: {
                    return new Mcts({
                        gameRules: game.rules,
                        planCount: player.planCount
                    });
                }
                case GamePlayerType.Alpha: {
                    return new Alpha({
                        gameRules: game.rules,
                        planCount: player.planCount,
                        model: game.createModel(),
                        modelName: player.modelName
                    });
                }
                case GamePlayerType.Human: {
                    return new Human(humanAction);
                }
            }
            return new Random(game.rules);
        }) as Agent[];
        const run = async () => {
            let gameState = game.rules.init();
            setGameState(gameState);
            let isDone = false;
            let rewards = [] as number[];
            for(let i = 1; !isDone && isRunningRef.current; i++) {
                setStatus(`Player ${gameState.playerIndex + 1}'s turn`);
                await sleep(10);
                const action = await agents[gameState.playerIndex].act();
                const gameStepResult = game.rules.step(
                    gameState, action
                );
                for (let i in agents) {
                    const agent = agents[i];
                    const index = +i;
                    if (index !== gameState.playerIndex) {
                        agent.step(action);
                    }
                }
                gameState = gameStepResult.state;
                setGameState(gameState);
                isDone = gameStepResult.done;
                rewards = gameStepResult.rewards;
            }
            switch (rewards[0]) {
                case (1): {
                    setStatus('Player 1 won');
                    break;
                }
                case (-1): {
                    setStatus('Player 2 won');
                    break;
                }
                default: {
                    setStatus(`It's a tie`);
                }
            }
            isRunningRef.current = false;
        };
        isRunningRef.current = true;
        run();
    }, [game, humanAction]);

    const stop = useCallback(() => {
        isRunningRef.current = false;
    }, []);

    return {
        gameState,
        status,
        onAction,
        start,
        stop
    };
};

export { useGame };