import {
  playContestAlpha,
  playSelfAlpha
} from './lib/play';
import Game from '../src/interfaces/game';
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
// import { GamePlayerType } from './interfaces/game-player';

const winRate = 0.55;

const trainGeneration = async (
  game: Game,
  generation: number
) => {
  const modelName = `alpha-${generation}`;
  const models = await getModels(game.name);
  if (models.includes(modelName)) {
    console.log(`${modelName} trained!`);
    return;
  }
  // const rules = game.rules;
  const planCount = game.players.find(
    player => player.type === GamePlayerType.Alpha
  )?.planCount || game.rules.actionsCount;
  const randomTurnsCount = game.randomTurnsCount;
  const previousGeneration = generation - 1;
  const previousModelName = `alpha-${previousGeneration}`;

  const previousModel = game.createModel();
  const model = game.createModel();

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
    const gameHistories = await playSelfAlpha({
      createWorker: () => game.createWorker(),
      model,
      gamesCount: 5000,
      averageTurns: 20,
      planCount,
      randomTurnsCount
    });

    for (let gameHistory of gameHistories) {
      modelHistories.push(gameHistory);
    }
    // console.log(modelHistories);
    await saveHistory(game.name, modelName, modelHistories);
  }

  // console.log(modelHistories);

  const loss = await model.train(modelHistories);
  if (loss > 1) {
    return false;
  }

  const gamesCount = 50;
  const contest = await playContestAlpha({
    createWorker: () => game.createWorker(),
    models: [
      model,
      previousModel
    ],
    gamesCount,
    averageTurns: 20,
    planCount,
    randomTurnsCount
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
      const success = await trainGeneration(game, i);
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

  const game = new GameClass();

  await trainAlpha(game);
};

export default run;