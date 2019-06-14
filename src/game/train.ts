import Random from './agents/random';
import PureMCTS from './agents/puremcts';
import { Alphazero, stateToInput} from './agents/alphazero';
import { get } from './model';
import { Env, getEquiData } from './env';
import * as tf from '@tensorflow/tfjs';
import { Swarm } from './agents/alphaswarm';

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
    count: number;
    constructor(setState) {
        this.setState = setState;
        this.count = 1;
    }
    async playGame(agent1, agent2, self = false, draw = false) {
        const players = [agent1, agent2];
        let done = false as any;
        let gameState = Env.init() as any;
        let player = gameState.player;
        let board = gameState.board;
        let reward;
        const history = [] as any[];
        const startPlayer = player;
        let time = Date.now();
        await sleep(1);
        if (draw) {
            this.setState({ player, board, done });
        }
        agent1.init();
        agent2.init();
        while (!done) {
            const current = players[player - 1];
            const enemy = players[2 - player];
            const action = await current.act(self);
            const newTime = Date.now();
            const secondsDelta = (newTime - time) / 1000;
            console.log(secondsDelta.toFixed(1));
            time = newTime;
            const entry = [gameState];
            const probs = current.getProbs();
            entry.push(probs);
            if (!self) {
                enemy.step(action);
            }
            history.push(entry);
            [ gameState, reward, done ] = Env.step(gameState, action);
            player = gameState.player;
            board = gameState.board;
            await sleep(1);
            if (draw) {
                const secondsPerTurn = (secondsDelta / this.count).toFixed(1);
                this.setState({ player, board, done, secondsPerTurn });
            }
        }
        if (player != startPlayer && reward) {
            reward *= -1;
        }
        return [ reward, history ]
    }
    async selfPlayParallel(model, count) {
        const games = new Array(count);
        this.count = count;
        const swarm = new Swarm(model, count);
        const agents = swarm.create();
        for (let i = 0; i < count; i++) {
            const agent = agents[i];
            const draw = i === 0;
            games[i] = this.playGame(agent, agent, true, draw);
        }
        const results = await Promise.all(games);
        return results;
    }
    async duelParallel(model1, model2, count) {
        this.count = count * 2;
        const games1 = new Array(count);
        const games2 = new Array(count);
        const swarm1 = new Swarm(model1, this.count);
        const swarm2 = new Swarm(model2, this.count);
        const agents1 = swarm1.create();
        const agents2 = swarm2.create();
        for (let i = 0; i < count; i++) {
            const player1 = agents1[i];
            const player2 = agents2[i];
            const draw = i === 0;
            games1[i] = this.playGame(player1, player2, false, draw);
        }
        for (let i = 0; i < count; i++) {
            const player1 = agents2[i + count];
            const player2 = agents1[i + count];
            games2[i] = this.playGame(player1, player2, false, false);
        }
        const results1 = Promise.all(games1);
        const results2 = Promise.all(games2);
        const results = await Promise.all([results1, results2]);
        return results;
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
            const result = await this.playGame(agent, pure, false, true);
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
            const result = await this.playGame(pure, agent, false, true);
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
        const trainingData = [] as any[];

        const GAMES_COUNT = 100;

        const results = await this.selfPlayParallel(model, GAMES_COUNT);
        console.log(results);

        const won = results.filter(result => result[0] === 1).length;
        const lost = results.filter(result => result[0] === -1).length;
        const ties = GAMES_COUNT - won - lost;

        console.log(`won: ${won}, ties: ${ties}, lost: ${lost}`);
        

        for (let result of results) {
            const [ reward, gameHistory ] = result;
            const newTrainingData = prepareTrainingData(reward, gameHistory);
            trainingData.push(...newTrainingData);
        }

        console.log(trainingData.length);
        // console.log(trainingData);

        const BATCH_SIZE = 64;
        // const TRAIN_BATCHES = 350;
        const TRAIN_BATCHES = 200;
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
            xs.dispose();
            tf.dispose(ys);
            console.log(trainingHistory.history.loss[0]);
        }

        const alpha = new Alphazero(model);
        const score = await this.validate(alpha);
        console.log(`score: ${score}`);
    }

    async duel(model1, model2) {
        console.log('Figth!');
        const DUEL_GAMES_COUNT = 10;

        let won = 0;
        let lost = 0;

        let results = await this.duelParallel(model1, model2, DUEL_GAMES_COUNT);

        const results1 = results[0];
        const results2 = results[1];
        console.log(results1, results2);

        won += results1.filter(result => result[0] === 1).length;
        lost += results1.filter(result => result[0] === -1).length;

        won += results2.filter(result => result[0] === -1).length;
        lost += results2.filter(result => result[0] === 1).length;

        console.log(`won: ${won}, lost: ${lost}`);
        const isBetter = won >= 1.2 * lost;
        return isBetter;
    }

    async initialPrepare() {
        console.log('begin initialPrepare...');
        const pure1 = new PureMCTS(5000);
        const pure2 = new PureMCTS(5000);
        const trainingData = [] as any[];


        for (let i = 0; i < 10; i++) {
            const result = await this.playGame(pure1, pure2);
            const [ reward, gameHistory ] = result;
            console.log(i, reward);
            const newTrainingData = prepareTrainingData(reward, gameHistory);
            trainingData.push(...newTrainingData);
        }

        // console.log(trainingData.length);
        console.log(trainingData);

        const init = {method: 'post'} as any;
        init.body = new FormData();

        init.body.append(
            'trainingData.json',
            new Blob(
                [JSON.stringify(trainingData)],
                {type: 'application/json'}),
            'trainingData.json');

        const response = await fetch('http://localhost:3080/models/data', init);
        console.log(response);
        return trainingData;
    }

    async initialLearn(){
        let model = await get();
        let trainingDataRequest = await fetch('http://localhost:3080/models/data/trainingData.json');
        const trainingData = await trainingDataRequest.json();

        const alpha = new Alphazero(model);

        // console.log(trainingData.length);
        console.log(trainingData);

        // const BATCH_SIZE = 64;
        // const BATCH_SIZE = 1;
        const BATCH_SIZE = 64;
        // const TRAIN_BATCHES = 100;
        const TRAIN_BATCHES = 350;

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
            xs.dispose();
            tf.dispose(ys);
            console.log(i, trainingHistory.history.loss[0]);
            // console.log(i, trainingHistory);
        }

        // const score = await this.validate(alpha);
        // console.log(`score: ${score}`);

        const saveResults =  await model.save('http://localhost:3080/models/initial');
        console.log(saveResults);

    }

    async run() {

        await this.initialPrepare();
        // await this.initialLearn();
        return;

        let current = await get();
        let saveResults =  await current.save('http://localhost:3080/models/current');
        console.log(saveResults);
        saveResults =  await current.save('http://localhost:3080/models/best');
        console.log(saveResults);
        let best = await tf.loadModel('http://localhost:3080/models/best/model.json');

        // let best = await tf.loadModel('http://localhost:3080/models/initial/model.json');
        // const isBetter = await this.duel(current, best);
        // console.log(isBetter);
        
        // const results = await this.playParallel(current, 100);
        // console.log(results);
        // return;

        let currentLevel = 0;
        let bestLevel = 0;
        let bestAge = 0;
        
        while ( bestAge < 6 ) {
            console.log(`level: ${currentLevel}, age: ${bestAge}`);
            await this.learn(current);
            currentLevel += 1;
            const isBetter = await this.duel(current, best);
            if (isBetter) {
                saveResults =  await current.save('http://localhost:3080/models/best');
                console.log(saveResults);
                best = await tf.loadModel('http://localhost:3080/models/best/model.json');
                bestLevel = currentLevel;
                console.log('new best!!');
            } else {
                if (bestLevel === 0) {
                    current = await get();
                    currentLevel = 0;
                    bestAge = 0;
                } else {
                    current = await get('http://localhost:3080/models/best/model.json');
                    currentLevel = bestLevel;
                    bestAge += 1;
                }
            }

        }
    }
}

export default Train;