import Random from './agents/random';
import PureMCTS from './agents/puremcts';
import { Alphazero, stateToInput} from './agents/alphazero';
import { get } from './model';
import { Env, getEquiData } from './env';
import * as tf from '@tensorflow/tfjs';

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const prepareTrainingData = (xReward, gameHistory) => {
    const trainingData = [] as any[];
    for (let entry of gameHistory) {
        const stateInitial = entry[0];
        const probsInitial = entry[1];
        const player = stateInitial.player;
        const boardInitial = stateInitial.board;
        let reward = xReward;
        if (player === 2 && reward) {
            reward *= -1;
        }
        for (let [ board, probs ] of getEquiData(boardInitial, probsInitial)){
            const state = { board, player };
            const input = stateToInput(state);
            const policy = probs;
            const value = [reward];
            trainingData.push([input, policy, value]);
        }
    }
    return trainingData;
};

class Train {
    setState;
    constructor(setState) {
        this.setState = setState;
    }
    async playGame(agent1, agent2, self = false) {
        const players = [agent1, agent2];
        let done = false as any;
        let gameState = Env.init() as any;
        let player = gameState.player;
        let board = gameState.board;
        let reward;
        const history = [] as any[];
        const startPlayer = player;
        await sleep(1);
        this.setState({ player, board, done });
        agent1.init();
        agent2.init();
        while (!done) {
            const current = players[player - 1];
            const enemy = players[2 - player];
            const action = await current.act(self);
            const entry = [gameState];
            if (self) {
                entry.push(current.getProbs());
            } else {
                enemy.step(action);
            }
            history.push(entry);
            [ gameState, reward, done ] = Env.step(gameState, action);
            player = gameState.player;
            board = gameState.board;
            this.setState({ player, board, done });
            await sleep(1);
        }
        if (player != startPlayer && reward) {
            reward *= -1;
        }
        return [ reward, history ]
    }

    async validate(agent) {
        console.log('validating...');
        const random = new Random();
        const pure = new PureMCTS();

        let won = 0;
        let lost = 0;
        let ties = 0;
        let score = 0;

        for (let i = 0; i < 1; i++) {
            const result = await this.playGame(agent, pure);
            const reward = result[0];
            console.log(i, reward);
            if (reward === 1) {
                won += 1;
            } else if (reward === 0) {
                ties += 1;
            } else {
                lost += 1;
                console.log(result);
            }
        }
        for (let i = 1; i < 2; i++) {
            const result = await this.playGame(pure, agent);
            const reward = result[0];
            console.log(i, -reward);
            if (reward === -1) {
                won += 1;
            } else if (reward === 0) {
                ties += 1;
            } else {
                lost += 1;
                console.log(result);
            }
        }
        score = -lost;
        console.log(`won: ${won}, ties: ${ties}, lost: ${lost}`);
        return score;
    }

    async learn(model) {
        console.log('learning...');
        const alpha = new Alphazero(model);
        const trainingData = [] as any[];


        for (let i = 0; i < 50; i++) {
            const result = await this.playGame(alpha, alpha, true);
            const [ reward, gameHistory ] = result;
            console.log(i, reward);
            const newTrainingData = prepareTrainingData(reward, gameHistory);
            trainingData.push(...newTrainingData);
        }

        console.log(trainingData.length);
        // console.log(trainingData);

        const BATCH_SIZE = 64;
        const TRAIN_BATCHES = 10;
        // const TRAIN_BATCHES = 1;

        for (let i = 0; i < TRAIN_BATCHES; i++) {
            trainingData.sort(() => 0.5 - Math.random());
            const batch = trainingData.slice(0, BATCH_SIZE);

            const xs = tf.tensor(batch.map(entry => entry[0]));
            const ys = [
                tf.tensor(batch.map(entry => entry[1])),
                tf.tensor(batch.map(entry => entry[2]))
            ];

            const trainingHistory = await model.fit(
                xs,
                ys,
                {
                    batchSize: BATCH_SIZE,
                    epochs: 1
                }
            );
            console.log(trainingHistory.history.loss[0]);
        }

        const score = await this.validate(alpha);
        console.log(`score: ${score}`);
    }

    async duel(model1, model2) {
        console.log('Figth!');
        const alpha1 = new Alphazero(model1, true);
        const alpha2 = new Alphazero(model2, true);
        let won = 0;
        let lost = 0;
        for (let i = 0; i < 10; i++) {
            const result = await this.playGame(alpha1, alpha2);
            const reward = result[0];
            console.log(i, reward);
            if (reward === 1) {
                won += 1;
            } else if (reward === -1) {
                lost += 1;
            }
        }
        for (let i = 10; i < 20; i++) {
            const result = await this.playGame(alpha2, alpha1);
            const reward = result[0];
            console.log(i, reward);
            if (reward === -1) {
                won += 1;
            } else if (reward === 1) {
                lost += 1;
            }
        }
        console.log(`won: ${won}, lost: ${lost}`)
        return won >= 1.4 * lost;
    }

    async run() {

        const pure = new PureMCTS(5000);

        let trainingData = [] as any[];

        for (let i = 0; i < 200; i++) {
            const result = await this.playGame(pure, pure, true);
            const [ reward, gameHistory ] = result;
            console.log(i, reward);
            const newTrainingData = prepareTrainingData(reward, gameHistory);
            trainingData.push(...newTrainingData);
        }

        console.log(trainingData);



        let current = await get();
        let saveResults =  await current.save('http://localhost:3000/models/current');
        console.log(saveResults);
        saveResults =  await current.save('http://localhost:3000/models/best');
        console.log(saveResults);
        let best = await tf.loadModel('http://localhost:3000/models/best/model.json');


        const BATCH_SIZE = 64;
        const TRAIN_BATCHES = 300;
        // const TRAIN_BATCHES = 1;

        for (let i = 0; i < TRAIN_BATCHES; i++) {
            trainingData.sort(() => 0.5 - Math.random());
            const batch = trainingData.slice(0, BATCH_SIZE);

            const xs = tf.tensor(batch.map(entry => entry[0]));
            const ys = [
                tf.tensor(batch.map(entry => entry[1])),
                tf.tensor(batch.map(entry => entry[2]))
            ];

            const trainingHistory = await current.fit(
                xs,
                ys,
                {
                    batchSize: BATCH_SIZE,
                    epochs: 1
                }
            );
            console.log(trainingHistory.history.loss[0], trainingHistory.history.dense_Dense1_acc[0]);
        }

        saveResults =  await current.save('http://localhost:3000/models/best');
        console.log(saveResults);
        return;
        let currentLevel = 0;
        let bestLevel = 0;
        
        while ( true ) {
            console.log(`level: ${currentLevel}`);
            await this.learn(current);
            currentLevel += 1;
            const isBetter = await this.duel(current, best);
            if (isBetter) {
                saveResults =  await current.save('http://localhost:3000/models/best');
                console.log(saveResults);
                best = await tf.loadModel('http://localhost:3000/models/best/model.json');
                bestLevel = currentLevel;
                console.log('new best!!');
            } else {
                if (bestLevel === 0) {
                    current = await get();
                    currentLevel = 0;
                } else {
                    current = await tf.loadModel('http://localhost:3000/models/current/model.json');
                    const LEARNING_RATE = 0.1;
                    const optimizer = tf.train.sgd(LEARNING_RATE);
                    current.compile({
                        optimizer: optimizer,
                        loss: [
                            'categoricalCrossentropy',
                            'meanSquaredError'
                        ],
                        metrics: ['accuracy']
                    });
                    currentLevel = bestLevel;
                }
            }

        }
    }
}

export default Train;