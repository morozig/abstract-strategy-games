import GameRules from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import PolicyAgent from '../interfaces/policy-agent';
import PolicyAction from '../interfaces/policy-action';
import { cpusCount, durationHR } from './helpers';
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

const getStates = (history: number[], rules: GameRules) => {
  const initial = rules.init();
  const states = [initial];
  let last = initial;
  for (let action of history) {
    const { state } = rules.step(last, action);
    states.push(state);
    last = state;
  }
  return states;
};

const play = async (
  gameRules: GameRules,
  agents: PolicyAgent[],
  name = ''
) => {
  let gameState = gameRules.init();
  for (let agent of agents) {
    agent.init();
  }
  let isDone = false;
  let rewards = [] as number[];
  const history = [] as PolicyAction[];
  for(let i = 1; !isDone; i++) {
    const policyAction = await agents[gameState.playerIndex].policyAct();
    const gameStepResult = gameRules.step(
      gameState, policyAction.action
    );
    for (let i in agents) {
      const agent = agents[i];
      const index = +i;
      if (index !== gameState.playerIndex) {
        agent.step(policyAction.action);
      }
    }
    history.push(policyAction);
    // console.log(`${name}:${i}`, gameState, policyAction.action);
    gameState = gameStepResult.state;
    isDone = gameStepResult.done;
    rewards = gameStepResult.rewards;
  }
  console.log(
    `game ${name} finished in ${history.length} moves`,
    rewards
  );
  return { rewards, history } as GameHistory;
};

interface PlaySelfAlphaOptions {
  model: AlphaModel;
  createWorker: () => Worker;
  gamesCount: number;
}

const playSelfAlpha = async (options: PlaySelfAlphaOptions) => {
  const {
    model,
    createWorker,
    gamesCount,
  } = options;

  // const size = 3;
  const size = await cpusCount();
  // console.log(await cpusCount());
  const concurrency = Math.ceil(gamesCount / size);

  console.log('size', size, 'concurrency', concurrency);

  const predictBatches = async (batches: TypedInput[][]) => {
    const batchIndices = batches.reduce(
      (indices, current) => indices.concat(
        indices[indices.length - 1] + current.length
      ),
      [0]
    );
    batchIndices.pop();

    const outputs = await model.predictBatch(batches.flat());
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
    thread.inputs().subscribe(async (inputBuffers) => {
      const inputs = ('send' in inputBuffers ?
          inputBuffers.send :
          inputBuffers
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
      thread.setOutputs(
        Transfer(policyBuffers, policyBuffers),
        Transfer(rewardBuffers, rewardBuffers)
      );
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
      const gameHistory = await worker.play(`${i}`);
      gameHistories.push(gameHistory);
    });
  }
  
  const gamesStart = new Date();
  console.log(
    `started ${gamesCount} games at`,
    gamesStart.toLocaleTimeString()
  );

  await pool.completed();
  await pool.terminate();

  const gamesEnd = new Date();
  console.log(
    `ended ${gamesCount} games in `,
    durationHR(gamesEnd.getTime() - gamesStart.getTime())
  );
  return gameHistories;
};

export {
  getStates,
  play,
  playSelfAlpha
};
