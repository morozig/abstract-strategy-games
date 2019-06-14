import { Env } from '../env';
import MCTS from '../../lib/mcts';
import * as tf from '@tensorflow/tfjs';

const stateToInput = (state) => {
    const board = state.board;
    const player = state.player;
    const input = [] as any[];

    for (let i = 0; i < 6; i++) {
        input.push([])
        for (let j = 0; j < 7; j++) {
            input[i].push([
                0, // our
                0, // enemy
                0, // color
            ]);
            if (board[i][j] === player) {
                input[i][j][0] = 1;
            } else if (board[i][j] === 3 - player) {
                input[i][j][1] = 1
            }
            if (player === 1) {
                input[i][j][2] = 1;
            }
        }
    }
   
    return input;
};

const predict = async (model, state) => {
    const input = stateToInput(state);
    const inputTensor = inputToTensor(input);

    const [ policyTensor, valueTensor ] = model.predict(inputTensor);
    const policyData = await policyTensor.data();
    const valueData = (await valueTensor.data())[0];

    inputTensor.dispose();
    policyTensor.dispose();
    valueTensor.dispose();

    return [policyData, valueData];
};

const inputToTensor = (input) => tf.tensor([input]);



const choose = probs => {
    const bound = Math.random();
    let current = 0;
    let currentAction = 1;
    for (let prob of probs) {
        current += Math.abs(prob);
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
        probs[index] = Math.abs(prob);
    }
    return probs;
};

class Agent {
    mcts: MCTS;
    model;
    probs;
    random;
    constructor(model, random = false) {
        this.model = model;
        this.random = random;
        this.init();
    }
    async policyValueFn(state) {
        const availables = Env.availables(state);
        const input = stateToInput(state);
        const inputTensor = inputToTensor(input);

        const [ policyTensor, valueTensor ] = this.model.predict(inputTensor);
        const policyData = await policyTensor.data();
        const valueData = (await valueTensor.data())[0];

        const policyRaw = availables.map(action => [
            action,
            policyData[action - 1]
        ]);
        const policySum = policyRaw.reduce((prev, cur) => prev + cur[1], 0);
        const policy = policyRaw.map(
            actionProb => [actionProb[0], actionProb[1] / policySum]
        );
        const value = Math.min(Math.max(valueData, -1), 1);
        return [ policy, value ];
    }
    async act(self = undefined) {
        await this.mcts.plan(400);
        let action;
        if (self || this.random) {
            const actionsProbs = this.mcts.getProbs();
            this.probs = actionsToProbs(actionsProbs); 
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
        this.mcts = new MCTS(Env, state => this.policyValueFn(state), {
            cPuct: 1
        });
    }
    getProbs() {
        return this.probs;
    }
}

export {
    Agent as Alphazero,
    stateToInput,
    predict
};