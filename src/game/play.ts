import { Alphazero, predict } from './agents/alphazero';
import { Env } from './env';
import * as tf from '@tensorflow/tfjs';
import Human from './agents/human';
import PureMCTS from './agents/puremcts';
import { get } from './model';

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

class Play {
    setState;
    channel;
    constructor(setState, channel) {
        this.setState = setState;
        this.channel = channel;
    }

    async run() {

        // let best = await tf.loadModel('http://localhost:3000/models/best/model.json');
        // let best = await get();
        let best = await get('http://localhost:3000/models/initial/model.json');

        const alpha = new Alphazero(best);
        const agent2 = new Human(2, this.channel);
        // const agent2 = new Human(2, this.channel);
        // const agent1 = alpha;
        // const agent1 = new Human(1, this.channel);
        const agent1 = new PureMCTS(10000);

        const players = [agent1, agent2];
        let done = false as any;
        let gameState = Env.init() as any;
        let player = gameState.player;
        let board = gameState.board;
        let reward;
        this.setState({ player, board, done });
        while (!done) {
            const timeBefore = +new Date();
            // const gameStates = new Array(256).fill(gameState);
            // const [policyData, valueData] = await predictBatch(best, gameStates);
            const [policyData, valueData] = await predict(best, gameState);
            const timeAfter = +new Date();
            console.log('time:', timeAfter - timeBefore);
            console.log(player, policyData, valueData);
            const current = players[player - 1];
            const enemy = players[2 - player];
            const action = await current.act();
            enemy.step(action);
            [ gameState, reward, done ] = Env.step(gameState, action);
            player = gameState.player;
            board = gameState.board;
            this.setState({ player, board, done });
        }
    }
}

export default Play;