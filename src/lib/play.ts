import GameRules from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import PolicyAgent from '../interfaces/policy-agent';
import PolicyAction from '../interfaces/policy-action';
import GameModel from '../interfaces/game-model';
import Alpha from '../agents/alpha';
import { durationHR } from './helpers';

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

export {
    play,
    playAlpha
};
