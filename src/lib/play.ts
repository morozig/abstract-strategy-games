import GameRules from '../interfaces/game-rules';
import Agent from '../interfaces/agent';
import GameHistory from '../interfaces/game-history';

const play = async (
    gameRules: GameRules,
    agents: Agent[],
    name = ''
) => {
    let gameState = gameRules.init();
    for (let agent of agents) {
        agent.init();
    }
    let isDone = false;
    let rewards = [] as number[];
    const history = [] as number[];
    for(let i = 1; !isDone; i++) {
        const action = await agents[gameState.playerIndex].act();
        const gameStepResult = gameRules.step(
            gameState, action
        );
        for (let i in agents) {
            const agent = agents[i];
            const index = +i;
            if (index !== gameState.playerIndex) {
                agent.step(action);
            }
        }
        history.push(action);
        console.log(`${name}:${i}`, gameState, action);
        gameState = gameStepResult.state;
        isDone = gameStepResult.done;
        rewards = gameStepResult.rewards;
    }
    console.log(
        `game ${name} finished in ${history.length} moves`,
        rewards
    );
    return {rewards, history} as GameHistory;
};

export default play;
