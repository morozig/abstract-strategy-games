import GameRules from '../interfaces/game-rules';
import GameModel from '../interfaces/game-model';
import GameHistory from '../interfaces/game-history';
import play from './play';
import Alpha from '../agents/alpha';
import { durationHR } from './helpers';

interface GamesOptions {
  model1: GameModel;
  model2: GameModel;
  gameRules: GameRules;
  gamesCount: number;
  switchSides?: boolean;
  planCount?: number;
  randomize?: boolean;
}

const playAlpha = async (options: GamesOptions) => {
  const gamePromises = [] as Promise<GameHistory>[];
  const alphaOptions = {
    gameRules: options.gameRules,
    planCount: options.planCount,
    randomize: options.randomize
  };
  for (let i = 0; i < options.gamesCount; i++) {
    const gamePromise = play(
        options.gameRules,
        [
            new Alpha({
              ...alphaOptions,
              model: options.model1
            }),
            new Alpha({
              ...alphaOptions,
              model: options.model2
            }),
        ],
        `${i + 1}`
    );
    gamePromises.push(gamePromise);
  }
  if (options.switchSides) {
    for (let i = options.gamesCount; i < options.gamesCount * 2; i++) {
      const gamePromise = play(
          options.gameRules,
          [
              new Alpha({
                ...alphaOptions,
                model: options.model2
              }),
              new Alpha({
                ...alphaOptions,
                model: options.model1
              }),
          ],
          `${i + 1}`
      );
      gamePromises.push(gamePromise);
    }
  }

  const gamesStart = new Date();
  console.log(
      `started ${gamePromises.length} games at`,
      gamesStart.toLocaleTimeString()
  );
  const gameHistories = await Promise.all(gamePromises);
  const gamesEnd = new Date();
  console.log(
      `ended ${gamePromises.length} games in `,
      durationHR(gamesEnd.getTime() - gamesStart.getTime())
  );
  return gameHistories;
};

export default playAlpha;