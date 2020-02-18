import GameRules from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import PolicyAgent from '../interfaces/policy-agent';
import PolicyAction from '../interfaces/policy-action';
import GameModel from '../interfaces/game-model';
import Alpha from '../agents/alpha';
import { durationHR, softMax } from './helpers';
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

interface CheckLevelOptions {
    model: GameModel;
    gameRules: GameRules;
    level: number;
    playerIndex: number;
    planCount?: number;
}

const checkLevel = async (options: CheckLevelOptions) => {
    const mistakes = [] as GameHistory[];
    if (options.level === 1) {
        return mistakes;
    }
    const alphaOptions = {
        gameRules: options.gameRules,
        planCount: options.planCount,
        model: options.model
    };
    for (let i = 1; i <= levelGamesCount; i++) {
        const pureMctsOptions = {
            gameRules: options.gameRules,
            planCount: Math.pow(2, options.level - 2),
            randomize: i !== 1
        };
        const agents = [
            new Mcts(pureMctsOptions),
            new Mcts(pureMctsOptions)
        ] as PolicyAgent[];
        agents[options.playerIndex] = new Alpha(alphaOptions);
        const { rewards, history } = await play(
            options.gameRules,
            agents,
            `${options.level}.${i}`
        );
        if (rewards[options.playerIndex] < 0) {
            mistakes.push({ rewards, history });
        }
    }
    return mistakes;
};

interface GetLevelOptions {
    model: GameModel;
    gameRules: GameRules;
    startLevel: number[];
    planCount?: number;
    maxLevel?: number;
}
  
const getLevel = async (options: GetLevelOptions) => {
    console.log(`GetLevel Started ${options.startLevel}`);
    const startLevel = options.startLevel;
    const maxLevel = options.maxLevel || 15;
    let player1Level = startLevel[0];
    let player2Level = startLevel[1];
    let mistakes = [] as GameHistory[];

    const startPlayer1LevelMistakes = await checkLevel({
        gameRules: options.gameRules,
        level: player1Level,
        model: options.model,
        playerIndex: 0,
        planCount: options.planCount
    });
    if (startPlayer1LevelMistakes.length) {
        player1Level -= 1;
    }

    const startPlayer2LevelMistakes = await checkLevel({
        gameRules: options.gameRules,
        level: player2Level,
        model: options.model,
        playerIndex: 1,
        planCount: options.planCount
    });
    if (startPlayer2LevelMistakes.length) {
        player2Level -= 1;
    }

    mistakes = mistakes.concat(
        startPlayer1LevelMistakes,
        startPlayer2LevelMistakes
    );

    if (mistakes.length) {
        return {
            level: [ player1Level, player2Level ],
            mistakes
        };
    } else {
        console.log('Successfully proved previos level!');
    }

    for (let level = player1Level + 1; true; level++) {
        if ( level > maxLevel ) {
            break;
        }
        
        const player1LevelMistakes = await checkLevel({
            gameRules: options.gameRules,
            level,
            model: options.model,
            playerIndex: 0,
            planCount: options.planCount
        });

        if (player1LevelMistakes.length) {
            mistakes = mistakes.concat(
                player1LevelMistakes
            );
            break;
        } else {
            console.log(`new player1 level: ${level}!`);
            player1Level = level;
        }
    }

    for (let level = player2Level + 1; true; level++) {
        if ( level > maxLevel ) {
            break;
        }
        
        const player2LevelMistakes = await checkLevel({
            gameRules: options.gameRules,
            level,
            model: options.model,
            playerIndex: 1,
            planCount: options.planCount
        });

        if (player2LevelMistakes.length) {
            mistakes = mistakes.concat(
                player2LevelMistakes
            );
            break;
        } else {
            console.log(`new player2 level: ${level}!`);
            player2Level = level;
        }
    }
    
    console.log(`GetLevel Finished ${[
        player1Level,
        player2Level
    ]} ${mistakes.length} mistakes`);
    return {
        level: [ player1Level, player2Level ],
        mistakes
    };
};

const fixGameHistory = (
    gameRules: GameRules,
    gameHistory: GameHistory
) => {
    const rewards = gameHistory.rewards.slice();
    const history = gameHistory.history.slice();
    const loserIndex = rewards.indexOf(-1);
    if (loserIndex === -1) {
        return {
            rewards,
            history
        } as GameHistory;
    }
    const fixedHistory = [] as PolicyAction[];

    let gameState = gameRules.init();

    for (let policyAction of history) {
        const availables = gameRules.availables(gameState);
        const fixedPolicy = gameState.playerIndex === loserIndex ?
            softMax(policyAction.policy, 1)
                .map(
                    (prob, index) =>
                        availables.includes(index + 1) ?
                            prob : 0
                )
                : policyAction.policy.slice();
        fixedHistory.push({
            action: policyAction.action,
            policy: fixedPolicy
        });
        const gameStepResult = gameRules.step(
            gameState, policyAction.action
        );
        gameState = gameStepResult.state;
    }

    return {
        rewards,
        history: fixedHistory
    } as GameHistory;
};

export {
    play,
    playAlpha,
    getLevel,
    fixGameHistory
};
