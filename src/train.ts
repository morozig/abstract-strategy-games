import {
    playAlpha,
    play
} from './lib/play';
import Game from '../src/interfaces/game';
import Mcts from './agents/mcts';
import GameHistory from './interfaces/game-history';
import {
    getModels,
    getHistories,
    loadHistory,
    saveHistory
} from './lib/api';

// import GameClass from './games/four-row';
import GameClass from './games/xos';
import { GamePlayerType } from './interfaces/game-player';

const winRate = 0.6;

const trainMcts = async (game: Game) => {
    const planCount = game.players.find(
        player => player.type === GamePlayerType.Mcts
    )?.planCount || game.rules.actionsCount; 
    const modelName = `mcts-${planCount}`;
    const models = await getModels(game.name);
    console.log(game.name, models);
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
            planCount,
            randomize: true,
            randomTurnsCount: game.randomTurnsCount
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
    // const trainSuccess = await model.train(modelHistories);
    await model.train(modelHistories);
    // if (trainSuccess) {
    await model.save(modelName);
    // }
};

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
    const planCount = game.players.find(
        player => player.type === GamePlayerType.Alpha
    )?.planCount || game.rules.actionsCount; 
    const previousGeneration = generation - 1;
    const previousModelName = `alpha-${previousGeneration}`;
    const previousModel = game.createModel(true);
    const model = game.createModel(true);

    if (previousGeneration > 0) {
        await previousModel.load(previousModelName);
        await model.load(previousModelName);
    }

    let modelHistories = [] as GameHistory[];
    const histories = await getHistories(game.name);
    if (histories.includes(modelName)) {
        const savedhistories = await loadHistory(
            game.name,
            modelName
        );
        for (let gameHistory of savedhistories) {
            modelHistories.push(gameHistory);
        }
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
        // console.log(modelHistories);
        await saveHistory(game.name, modelName, modelHistories);
    }
    console.log(modelHistories);
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
    const player1Won = contest
        .slice(0, gamesCount)
        .filter(({ rewards }) => rewards[0] === 1)
        .length;
    const player1Lost = contest
        .slice(0, gamesCount)
        .filter(({ rewards }) => rewards[0] === -1)
        .length;
    const player2Won = contest
        .slice(gamesCount)
        .filter(({ rewards }) => rewards[1] === 1)
        .length;
    const player2Lost = contest
        .slice(gamesCount)
        .filter(({ rewards }) => rewards[1] === -1)
        .length;
    const modelScore = (player1Won + player2Won) / (
        (player1Won + player1Lost + player2Won + player2Lost) || 1
    );
    console.log(
        'player1Won', player1Won,
        'player1Lost', player1Lost,
        'player2Won', player2Won,
        'player2Lost', player2Lost
    );
    console.log(`score: ${modelScore.toFixed(2)}`);
        
    if (modelScore >= winRate) {
        await model.save(modelName);
    }
    return modelScore >= winRate;
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
        const maxTries = 30;
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

    const game = new GameClass(5, 5, 4);

    // console.log(trainMcts);
    console.log(trainAlpha);
    // await trainAlpha(game);
    await trainMcts(game);
};

export default run;