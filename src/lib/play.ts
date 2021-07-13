import GameHistory from '../interfaces/game-history';
import { cpusCount } from './helpers';
import AlphaModel from './alpha-model';
import {
  spawn,
  Pool,
  Worker,
  Transfer
} from 'threads';
import { PlayWorkerType } from './play-worker';
import Batcher from './batcher';
import { TypedInput } from './alpha-network';
import ProgressBar from './progress-bar';
import {
  getHistories,
  loadHistory,
  saveHistory
} from '../lib/api';

interface PlaySelfAlphaOptions {
  model: AlphaModel;
  modelName: string;
  createWorker: () => Worker;
  gamesCount: number;
  averageTurns: number;
  planCount: number;
  randomTurnsCount?: number;
}

const playSelfAlpha = async (options: PlaySelfAlphaOptions) => {
  const {
    model,
    modelName,
    createWorker,
    averageTurns,
    planCount,
    randomTurnsCount
  } = options;

  const gameHistories = [] as GameHistory[];
  const histories = await getHistories(model.gameName);
  if (histories.includes(modelName)) {
    const savedhistories = await loadHistory(
      model.gameName,
      modelName
    );
    for (let gameHistory of savedhistories) {
      gameHistories.push(gameHistory);
    }
  }

  const gamesCount = options.gamesCount - gameHistories.length;
  if (!gamesCount) {
    return gameHistories;
  }

  // const size = 3;
  const size = await cpusCount();
  // console.log(await cpusCount());
  const concurrency = Math.min(
    Math.ceil(gamesCount / size),
    150
  );

  console.log('size', size, 'concurrency', concurrency);

  const progressBar = new ProgressBar(
    ''.concat(
      '[:bar] :percent | ETA: :eta | ',
      ':gamesComplete/:gamesCount games | ',
      'speed :speed | :elapsed',
    ),
    {
      total: gamesCount * averageTurns * planCount * 0.9,
      tokens: {
        gamesComplete: 0,
        gamesCount,
        speed: 'N/A'
      }
    }
  );

  const predictBatches = async (batches: TypedInput[][]) => {
    const batchIndices = batches.reduce(
      (indices, current) => indices.concat(
        indices[indices.length - 1] + current.length
      ),
      [0]
    );
    batchIndices.pop();

    const startTime = new Date().getTime();
    const outputs = await model.predictBatch(batches.flat());
    const endTime = new Date().getTime();
    const speed = `${((endTime - startTime) / outputs.length).toFixed(2)} ms`;
    progressBar.update(
      curr => curr + outputs.length,
      tokens => ({speed})
    );
    const batchOutputs = batchIndices.map(
      (batchIndex, i, arr) => outputs.slice(batchIndex, arr[i + 1])
    );
    return batchOutputs;
  };
  const batcher = new Batcher(
    (batches: TypedInput[][]) => predictBatches(batches),
    size,
    50
  );

  const spawnWorker = async () => {
    const thread = await spawn<PlayWorkerType>(createWorker());
    thread.inputs().subscribe(async (inputsRequest) => {
      const inputs = ('send' in inputsRequest.inputs ?
          inputsRequest.inputs.send :
          inputsRequest.inputs
        ).map(
        inputBuffer => new Float32Array(inputBuffer)
      );
      const outputs = await batcher.call(inputs);
      const policyBuffers = outputs.map(
        output => output[0].buffer
      );
      const rewardBuffers = outputs.map(
        output => output[1].buffer
      );
      thread.setOutputs({
        policies: Transfer(policyBuffers, policyBuffers),
        rewards: Transfer(rewardBuffers, rewardBuffers)
      });
    })
    return thread;
  }

  const pool = Pool(spawnWorker, {
    concurrency,
    size
  });

  for (let i = 0; i < gamesCount; i++) {
    pool.queue(async worker => {
      const gameHistory = await worker.play({
        gameName: `${i}`,
        planCount,
        randomTurnsCount
      });
      const turnsDelta = Math.max(
        averageTurns - gameHistory.history.length,
        0
      );
      progressBar.update(
        curr => curr + turnsDelta * planCount,
        tokens => ({
          gamesComplete: tokens.gamesComplete + 1
        })
      ); 
      gameHistories.push(gameHistory);
      if (gameHistories.length % 100 === 0) {
        await saveHistory(
          model.gameName,
          modelName,
          gameHistories
        )
      }
    });
  }
  
  // const gamesStart = new Date();
  // console.log(
  //   `started ${gamesCount} games at`,
  //   gamesStart.toLocaleTimeString()
  // );
  progressBar.start();

  await pool.completed();
  await pool.terminate();

  // const gamesEnd = new Date();
  // console.log(
  //   `ended ${gamesCount} games in `,
  //   durationHR(gamesEnd.getTime() - gamesStart.getTime())
  // );
  progressBar.update(gamesCount * averageTurns * planCount * 0.9);
  progressBar.stop();
  await saveHistory(
    model.gameName,
    modelName,
    gameHistories
  )
  return gameHistories;
};

interface PlayContestAlphaOptions {
  models: AlphaModel[];
  createWorker: () => Worker;
  gamesCount: number;
  averageTurns: number;
  planCount: number;
  randomTurnsCount?: number;
}

const playContestAlpha = async (options: PlayContestAlphaOptions) => {
  const {
    models,
    createWorker,
    gamesCount,
    averageTurns,
    planCount,
    randomTurnsCount
  } = options;

  // const size = 3;
  const size = await cpusCount();
  // console.log(await cpusCount());
  const concurrency = Math.min(
    Math.ceil(gamesCount * 2 / size),
    150
  );

  console.log('size', size, 'concurrency', concurrency);

  const progressBar = new ProgressBar(
    ''.concat(
      '[:bar] :percent | ETA: :eta | ',
      ':gamesComplete/:gamesCount games | ',
      'speed :speed | :elapsed',
    ),
    {
      total: 2 * gamesCount * averageTurns * planCount * 0.9,
      tokens: {
        gamesComplete: 0,
        gamesCount: 2 * gamesCount,
        speed: 'N/A'
      }
    }
  );

  const predictBatches = async (
    batches: TypedInput[][],
    model: AlphaModel
  ) => {
    const batchIndices = batches.reduce(
      (indices, current) => indices.concat(
        indices[indices.length - 1] + current.length
      ),
      [0]
    );
    batchIndices.pop();

    const startTime = new Date().getTime();
    const outputs = await model.predictBatch(batches.flat());
    const endTime = new Date().getTime();
    const speed = `${((endTime - startTime) / outputs.length).toFixed(2)} ms`;
    progressBar.update(
      curr => curr + outputs.length,
      tokens => ({speed})
    );
    const batchOutputs = batchIndices.map(
      (batchIndex, i, arr) => outputs.slice(batchIndex, arr[i + 1])
    );
    return batchOutputs;
  };

  const modelsBatchers = models.map(
    model => new Batcher(
      (batches: TypedInput[][]) => predictBatches(batches, model),
      size,
      50
    )
  );

  const spawnWorker = async () => {
    const thread = await spawn<PlayWorkerType>(createWorker());
    thread.inputs().subscribe(async (inputsRequest) => {
      const inputs = ('send' in inputsRequest.inputs ?
          inputsRequest.inputs.send :
          inputsRequest.inputs
        ).map(
        inputBuffer => new Float32Array(inputBuffer)
      );
      const modelIndex = inputsRequest.modelIndex || 0;
      const outputs = await modelsBatchers[modelIndex].call(inputs);
      const policyBuffers = outputs.map(
        output => output[0].buffer
      );
      const rewardBuffers = outputs.map(
        output => output[1].buffer
      );
      thread.setOutputs({
        policies: Transfer(policyBuffers, policyBuffers),
        rewards: Transfer(rewardBuffers, rewardBuffers),
        modelIndex
      });
    })
    return thread;
  }

  const pool = Pool(spawnWorker, {
    concurrency,
    size
  });

  const gameHistories = [] as GameHistory[];
  for (let i = 0; i < gamesCount; i++) {
    pool.queue(async worker => {
      const gameHistory = await worker.play({
        gameName: `${i}`,
        modelsIndices: [0, 1],
        planCount,
        randomTurnsCount
      });
      const turnsDelta = Math.max(
        averageTurns - gameHistory.history.length,
        0
      );
      progressBar.update(
        curr => curr + turnsDelta * planCount,
        tokens => ({
          gamesComplete: tokens.gamesComplete + 1
        })
      ); 
      gameHistories.push(gameHistory);
    });
  }
  for (let i = gamesCount; i < gamesCount * 2; i++) {
    pool.queue(async worker => {
      const gameHistory = await worker.play({
        gameName: `${i}`,
        modelsIndices: [1, 0],
        planCount,
        randomTurnsCount
      });
      const turnsDelta = Math.max(
        averageTurns - gameHistory.history.length,
        0
      );
      progressBar.update(
        curr => curr + turnsDelta * planCount,
        tokens => ({
          gamesComplete: tokens.gamesComplete + 1
        })
      ); 
      gameHistories.push(gameHistory);
    });
  }
  
  // const gamesStart = new Date();
  // console.log(
  //   `started ${gamesCount} games at`,
  //   gamesStart.toLocaleTimeString()
  // );
  progressBar.start();

  await pool.completed();
  await pool.terminate();

  // const gamesEnd = new Date();
  // console.log(
  //   `ended ${gamesCount} games in `,
  //   durationHR(gamesEnd.getTime() - gamesStart.getTime())
  // );
  progressBar.update(2 * gamesCount * averageTurns * planCount * 0.9);
  progressBar.stop();
  return gameHistories;
};

export {
  playSelfAlpha,
  playContestAlpha
};
