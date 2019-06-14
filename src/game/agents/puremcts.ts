import { Env } from '../env';
import MCTS from '../../lib/mcts';

const rollout = (state) => {
    let tempState = state;
    let done;
    let reward;
    while (!done) {
        const availables = Env.availables(tempState);
        const index = Math.floor(Math.random() * availables.length);
        const action = availables[index];
        [ tempState, reward, done ] = Env.step(tempState, action);
    }
    if (state.player != tempState.player) {
        reward *= -1;
    }
    return reward as number;
};

const rolloutPolicyValueFn = (state) => {
    const availables = Env.availables(state);
    const count = availables.length;
    const policy = availables.map(action => [action, 1 / count]);
    const value = rollout(state);
    return [policy, value];
};

const badPolicyValueFn = (state) => {
    const availables = Env.availables(state);
    const count = availables.length;
    const policy = availables.map(action => [action, 1 / count]);

    const value = Math.floor(Math.random() * 3 - 1);
    return [policy, value];
};

const choose = probs => {
    const bound = Math.random();
    let current = 0;
    let currentAction = 1;
    for (let prob of probs) {
        current += prob;
        if (current >= bound) {
            break;
        }
        currentAction += 1;
    }
    return currentAction;
};

const softmax = (vals, temp) => {
    const sumExp = vals
        .map(val => Math.exp(val / temp))
        .reduce((a, b) => a + b);
    return vals
        .map(val => Math.exp(val / temp) / sumExp);
};

const actionsToSoftMaxedProbs = actionsProbs => {
    const probs = Array(7).fill(0);
    const softmaxedProbs = softmax(
        actionsProbs.map(actionProb => actionProb[1]),
        0.01
    );
    const softActionsProbs = actionsProbs
        .map((actionProb, index) => [ actionProb[0], softmaxedProbs[index]]);

    for (let [ action, prob ] of softActionsProbs){
        const index = action - 1;
        probs[index] = prob;
    }
    return probs;
};

const actionsToProbs = actionsProbs => {
    const probs = Array(7).fill(0);
    for (let [ action, prob ] of actionsProbs){
        const index = action - 1;
        probs[index] = prob;
    }
    return probs;
};


class Agent {
    mcts: MCTS;
    planNumber;
    probs;
    constructor(planNumber = 5000) {
        this.planNumber = planNumber;
        this.init();
    }
    async act(self = undefined) {
        await this.mcts.plan(this.planNumber);
        let action;
        const actionsProbs = this.mcts.getProbs();
        this.probs = actionsToProbs(actionsProbs); 
        if (self) {
            const probs = actionsToSoftMaxedProbs(actionsProbs);
            action = choose(probs);
        } else {
            action = this.mcts.getAction();
        }
        this.mcts.step(action);
        return action;
    }
    step(action: number) {
        this.mcts.step(action);
    }
    init() {
        this.mcts = new MCTS(Env, rolloutPolicyValueFn);
    }
    getProbs() {
        return this.probs;
    }
}

export default Agent;