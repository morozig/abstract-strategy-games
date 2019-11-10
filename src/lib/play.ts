import GameRules from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import PolicyAgent from '../interfaces/policy-agent';
import PolicyAction from '../interfaces/policy-action';

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

export default play;
