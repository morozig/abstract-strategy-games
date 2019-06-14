import { Env } from '../env';
import MCTS from '../../lib/mcts';
import * as tf from '@tensorflow/tfjs';
import Channel from '../../lib/channel';

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

class Request {
    input;
    channel: Channel;
    constructor(input) {
        this.input = input;
        this.channel = new Channel();
    }
}

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
        0.05
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
    predict;
    mcts: MCTS;
    probs;
    random;
    constructor(predict) {
        this.predict = predict;
        this.init();
    }
    async policyValueFn(state) {
        const availables = Env.availables(state);
        const input = stateToInput(state);

        const [ policyData, valueData ] = await this.predict(input);

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
    async act() {
        await this.mcts.plan(400);
        const actionsProbs = this.mcts.getProbs();
        this.probs = actionsToProbs(actionsProbs); 
        const probs = actionsToSoftMaxedProbs(actionsProbs);
        const action = choose(probs);
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

class Swarm {
    model;
    count;
    agents;
    requests: Request[];
    interval;
    time;
    full: Channel;
    lastCount;
    constructor(model, count) {
        this.model = model;
        this.count = count;
        this.agents = new Array(count);
        this.interval = 10 + count;
        this.requests = [];
        this.full = new Channel();
        this.lastCount = count;
    }
    predict(input) {
        const request = new Request(input);
        this.requests.push(request);
        if (this.requests.length == 1) {
            this.predictBatch();
        }
        if (this.requests.length === this.lastCount) {
            this.full.set(1);
        }
        return request.channel.get();
    }
    async predictBatch() {
        await Promise.race([
            this.full.get(),
            sleep(this.interval)
        ]);
        this.time = Date.now();
        const requests = this.requests.slice();
        this.lastCount = requests.length;
        this.requests = [];
        const inputs = requests.map(request => request.input);
        const inputsTensor = tf.tensor(inputs);
        const [ policyTensor, valueTensor ] = this.model.predict(inputsTensor);
        const policyData = await policyTensor.data();
        const valueData = await valueTensor.data();

        inputsTensor.dispose();
        policyTensor.dispose();
        valueTensor.dispose();

        this.interval = Math.ceil( ( Date.now() - this.time ) * 0.9 );
        for (let i = 0; i < requests.length; i++) {
            const request = requests[i];
            const policy = policyData.slice(i * 7, ( i + 1 ) * 7);
            const value = valueData[i];
            request.channel.set([ policy, value ]);
        }
    }

    create() {
        for (let i = 0; i < this.count; i++) {
            this.agents[i] = new Agent(input => this.predict(input));
        }
        return this.agents;
    }
}

export {
    Swarm
};