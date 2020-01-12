import GameRules from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import PolicyAgent from '../interfaces/policy-agent';
import PolicyAction from '../interfaces/policy-action';
import GameModel from '../interfaces/game-model';
import Alpha from '../agents/alpha';
import { durationHR } from './helpers';
import Mcts from '../agents/mcts';

const levelGamesCount = 5;

const play = async (
    gameRules: GameRules,
    agents: PolicyAgent[],
    name = ''
) => {
    let gameState = gameRules.init();
    for (let agent of agents) {
        agent.init();
    }
    let isDone = false;
    let rewards = [] as number[];
    const history = [] as PolicyAction[];
    for(let i = 1; !isDone; i++) {
        const policyAction = await agents[gameState.playerIndex].policyAct();
        const gameStepResult = gameRules.step(
            gameState, policyAction.action
        );
        for (let i in agents) {
            const agent = agents[i];
            const index = +i;
            if (index !== gameState.playerIndex) {
                agent.step(policyAction.action);
            }
        }
        history.push(policyAction);
        console.log(`${name}:${i}`, gameState, policyAction.action);
        gameState = gameStepResult.state;
        isDone = gameStepResult.done;
        rewards = gameStepResult.rewards;
    }
    console.log(
        `game ${name} finished in ${history.length} moves`,
        rewards
    );
    return { rewards, history } as GameHistory;
};


interface PlayAlphaOptions {
    model1: GameModel;
    model2: GameModel;
    gameRules: GameRules;
    gamesCount: number;
    switchSides?: boolean;
    planCount?: number;
    randomize?: boolean;
}
  
const playAlpha = async (options: PlayAlphaOptions) => {
    const gamePromises = [] as Promise<GameHistory>[];
    const alphaOptions = {
        gameRules: options.gameRules,
        planCount: options.planCount,
        randomize: options.randomize
    };
    for (let i = 0; i < options.gamesCount; i++) {
        const gamePromise = play(
            options.gameRules,
            [
                new Alpha({
                ...alphaOptions,
                model: options.model1
                }),
                new Alpha({
                ...alphaOptions,
                model: options.model2
                }),
            ],
            `${i + 1}`
        );
        gamePromises.push(gamePromise);
    }
    if (options.switchSides) {
        for (let i = options.gamesCount; i < options.gamesCount * 2; i++) {
        const gamePromise = play(
            options.gameRules,
            [
                new Alpha({
                    ...alphaOptions,
                    model: options.model2
                }),
                new Alpha({
                    ...alphaOptions,
                    model: options.model1
                }),
            ],
            `${i + 1}`
        );
        gamePromises.push(gamePromise);
        }
    }

    const gamesStart = new Date();
    console.log(
        `started ${gamePromises.length} games at`,
        gamesStart.toLocaleTimeString()
    );
    const gameHistories = await Promise.all(gamePromises);
    const gamesEnd = new Date();
    console.log(
        `ended ${gamePromises.length} games in `,
        durationHR(gamesEnd.getTime() - gamesStart.getTime())
    );
    return gameHistories;
};

interface GetLevelOptions {
    model: GameModel;
    gameRules: GameRules;
    startLevel: number[];
    planCount?: number;
    alwaysTryAllPlayers?: boolean;
    maxLevel?: number;
}
  
const getLevel = async (options: GetLevelOptions) => {
    console.log(`GetLevel Started ${options.startLevel}`);
    const alphaOptions = {
        gameRules: options.gameRules,
        planCount: options.planCount,
        model: options.model
    };
    const startLevel = options.startLevel;
    const maxLevel = options.maxLevel || 15;
    let player1Level = startLevel[0];
    let player2Level = startLevel[1];
    const alwaysTryAllPlayers = options.alwaysTryAllPlayers || false;
    console.log(`Current player1 level: ${player1Level}`);
    for (let level = player1Level; true; level++) {
        if (level === 1) {
            continue;
        }
        if ( level > maxLevel ) {
            break;
        }
        let score = 0;
        for (let i = 1; i <= levelGamesCount; i++) {
            const pureMctsOptions = {
                gameRules: options.gameRules,
                planCount: Math.pow(2, level - 2),
                randomize: i !== 1
            };
            const player1 = new Alpha(alphaOptions);
            const player2 = new Mcts(pureMctsOptions);
            const agents = [player1, player2];
            const { rewards } = await play(
                options.gameRules,
                agents,
                `${level}.${i}`
            );
            if (rewards[0] < 1) {
                break;
            } else {
                score += 1;
            }
        }
        if (score < levelGamesCount) {
            if (level === startLevel[0]) {
                player1Level -= 1;
            }
            break;
        } else {
            console.log(`new player1 level: ${level}!`);
            player1Level = level;
        }
    }
    if (player1Level < startLevel[0] && !alwaysTryAllPlayers) {
        return [ player1Level, player2Level ];
    }
    console.log(`Current player2 level: ${player2Level}`);
    for (let level = player2Level; true; level++) {
        if (level === 1) {
            continue;
        }
        if ( level > maxLevel ) {
            break;
        }
        let score = 0;
        for (let i = 1; i <= levelGamesCount; i++) {
            const pureMctsOptions = {
                gameRules: options.gameRules,
                planCount: Math.pow(2, level - 2),
                randomize: i !== 1
            };
            const player1 = new Mcts(pureMctsOptions);
            const player2 = new Alpha(alphaOptions);
            const agents = [player1, player2];
            const { history } = await play(
                options.gameRules,
                agents,
                `${level}.${i}`
            );
            if (history.length < 6) {
                break;
            } else {
                score += 1;
            }
        }
        if (score < levelGamesCount) {
            if (level === startLevel[1]) {
                player2Level -= 1;
            }
            break;
        } else {
            console.log(`new player2 level: ${level}!`);
            player2Level = level;
        }
    }
    console.log(`GetLevel Finished ${[ player1Level, player2Level ]}`);
    return [ player1Level, player2Level ];
};

export {
    play,
    playAlpha,
    getLevel
};
