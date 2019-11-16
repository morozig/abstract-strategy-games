import play from './lib/play';
import Game from '../src/interfaces/game';

import GameClass from './games/four-row';
import Random from './agents/random';
import GameHistory from './interfaces/game-history';
import {
    getModels, getHistories, loadHistory, saveHistory
} from './lib/api';
import Mcts from './agents/mcts';
import playAlpha from './lib/play-alpha';

const trainMcts = async (game: Game) => {
    const modelName = 'mcts-5000';
    const models = await getModels(game.name);
    if (models.includes(modelName)) {
        console.log(`${modelName} trained!`);
        return;
    }
    const modelHistories = [] as GameHistory[];
    const histories = await getHistories(game.name);
    if (histories.includes(modelName)) {
        const savedhistories = await loadHistory(
            game.name,
            modelName
        );
        for (let gameHistory of savedhistories) {
            modelHistories.push(gameHistory);
        }
        console.log(modelHistories);
    } else {
        const rules = game.rules;
        const pureMctsOptions = {
            gameRules: rules,
            planCount: 5000
        };
        const player1 = new Mcts(pureMctsOptions);
        const player2 = new Mcts(pureMctsOptions);
        const agents = [player1, player2];

        for (let i = 0; i < 100; i++) {
            const gameHistory = await play(
                rules,
                agents,
                `${i + 1}`
            );
            modelHistories.push(gameHistory);
        }
        console.log(modelHistories);
        await saveHistory(game.name, modelName, modelHistories);
    }
    const model = game.createModel();
    const trainSuccess = await model.train(modelHistories);
    await model.save(modelName);
};

const trainGeneration = async (game: Game, generation: number) => {
    const modelName = `alpha-${generation}`;
    const models = await getModels(game.name);
    if (models.includes(modelName)) {
        console.log(`${modelName} trained!`);
        return;
    }
    const rules = game.rules;
    const previousGeneration = generation - 1;
    const previousModelName = `alpha-${previousGeneration}`;
    const previousModel = game.createModel(true);
    const model = game.createModel(true);
    if (previousGeneration > 0) {
        await previousModel.load(previousModelName);
        await model.load(previousModelName);
    }

    const modelHistories = [] as GameHistory[];
    const histories = await getHistories(game.name);
    if (histories.includes(modelName)) {
        const savedhistories = await loadHistory(
            game.name,
            modelName
        );
        for (let gameHistory of savedhistories) {
            modelHistories.push(gameHistory);
        }
        console.log(modelHistories);
    } else {
        const gameHistories = await playAlpha({
            gameRules: rules,
            model1: model,
            model2: model,
            gamesCount: 100,
            planCount: 300,
            randomize: true
        });

        for (let gameHistory of gameHistories) {
            modelHistories.push(gameHistory);
        }
        console.log(modelHistories);
        await saveHistory(game.name, modelName, modelHistories);
    }
    await model.train(modelHistories);

    const gamesCount = 5;
    const contest = await playAlpha({
        gameRules: rules,
        model1: model,
        model2: previousModel,
        gamesCount: gamesCount,
        switchSides: true,
        planCount: 300,
        randomize: true
    });
    const modelScore =
        contest
            .slice(0, gamesCount)
            .reduce((score, history) => score + history.rewards[0], 0)
        +
        contest
            .slice(gamesCount)
            .reduce((score, history) => score + history.rewards[1], 0);
    console.log(`score: ${modelScore}`);
    if (modelScore > gamesCount + 1) {
        await model.save(modelName);
    }
    return modelScore > gamesCount + 1;
};

const trainAlpha = async (game: Game) => {
    const models = await getModels(game.name);
    const generations = models
        .filter(model => model.includes('alpha'))
        .map(model => model.replace('alpha-', ''))
        .map(model => parseInt(model));
    const lastGeneration = generations.length > 0 ?
        Math.max(...generations) + 1 : 1;
    for (let i = lastGeneration; true; i++) {
        let tries = 0;
        const triesCount = 10;
        while (tries <= triesCount) {
            tries += 1;
            console.log(`train ${i}:${tries}`);
            const success = await trainGeneration(game, i);
            if (success) {
                break;
            }
        }
        if (tries <= triesCount) {
            console.log(`new generation ${i + 1}!`);
        } else {
            break;
        }
    }
};


const run = async () => {
    console.log('train started');

    const game = new GameClass();

    // await trainMcts(game);
    await trainAlpha(game);
};

export default run;