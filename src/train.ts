import {
    playAlpha,
    getLevel
} from './lib/play';
import Game from '../src/interfaces/game';

// import GameClass from './games/four-row';
import GameClass from './games/xos';

import GameHistory from './interfaces/game-history';
import {
    getModels,
    getHistories,
    loadHistory,
    saveHistory,
    loadResult,
    saveResult
} from './lib/api';

const planCount = 50;
const maxLevel = 15;
const useLevels = true;

const trainGeneration = async (
    game: Game,
    generation: number,
    improve?: boolean
) => {
    const modelName = `alpha-${generation}`;
    const models = await getModels(game.name);
    if (models.includes(modelName)) {
        console.log(`${modelName} trained!`);
        return;
    }
    const rules = game.rules;
    const previousGeneration = generation - 1;
    const previousModelName = `alpha-${previousGeneration}`;
    let previousModelLevel = [1, 1];
    const previousModel = game.createModel(true);
    const model = game.createModel(true);

    if (previousGeneration > 0) {
        await previousModel.load(previousModelName);
        await model.load(previousModelName);

        if (useLevels) {
            const previousModelResult = await loadResult(
                game.name,
                previousModelName
            );
            if (previousModelResult.level) {
                previousModelLevel[0] = previousModelResult.level[0];
                previousModelLevel[1] = previousModelResult.level[1];
            }
        }
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
            planCount,
            randomize: true
        });

        for (let gameHistory of gameHistories) {
            modelHistories.push(gameHistory);
        }
        console.log(modelHistories);
        await saveHistory(game.name, modelName, modelHistories);
    }
    await model.train(modelHistories, {improve});

    const gamesCount = 25;
    const contest = await playAlpha({
        gameRules: rules,
        model1: model,
        model2: previousModel,
        gamesCount: gamesCount,
        switchSides: true,
        planCount,
        randomize: true
    });
    const points = contest
        .slice(0, gamesCount)
        .filter(({ rewards }) => rewards[0] === 1)
        .length
        + contest
        .slice(gamesCount)
        .filter(({ rewards }) => rewards[1] !== -1)
        .length;
    const modelScore = points / (2 * gamesCount );
    console.log(`score: ${modelScore}`);
        
    if (modelScore >= 0.6) {
        if (useLevels) {
            const modelLevel = await getLevel({
                gameRules: rules,
                model,
                startLevel: previousModelLevel,
                planCount,
                maxLevel
            });
            
            console.log(`previous level: ${previousModelLevel}`);
            console.log(`current level: ${modelLevel}`);
        
            if (modelLevel[0] < previousModelLevel[0] || 
                modelLevel[1] < previousModelLevel[1]
            ) {
                console.log('Failed to beat previous level');
                return false;
            }
            await saveResult(
                game.name,
                modelName,
                {
                    level: modelLevel
                }
            );
        }
        await model.save(modelName);
    }
    return modelScore >= 0.6;
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
        let tries = 1;
        const maxTries = 20;
        while (tries <= maxTries) {
            console.log(`train ${i}:${tries}`);
            const improve = tries > maxTries / 2;
            const success = await trainGeneration(game, i, improve);
            tries += 1;
            if (success) {
                break;
            }
        }
        if (tries <= maxTries) {
            console.log(`new generation ${i + 1}!`);
        } else {
            break;
        }
    }
};


const run = async () => {
    console.log('train started');

    const game = new GameClass(4, 4, 4);

    // console.log(trainMcts);
    // console.log(trainAlpha);
    await trainAlpha(game);
};

export default run;