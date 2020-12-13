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
    console.log(`${name}:${i}`, gameState, policyAction.action);
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
  worker: Worker;
  gamesCount: number;
}

const playSelfAlpha = async (options: PlaySelfAlphaOptions) => {
  const {
    model,
    worker,
    gamesCount,
  } = options;

  const size = await cpusCount();
  const concurrency = Math.min(
    Math.floor(options.gamesCount / size) || 1,
    100
  );

  const batcher = new Batcher(
    (inputs: TypedInput[]) => model.predictBatch(inputs),
    size * concurrency,
    10
  );

  const spawnWorker = async () => {
    const thread = await spawn<PlayWorkerType>(worker);
    thread.inputs().subscribe(async (inputBuffer) => {
      const input = new Float32Array(inputBuffer as TypedInput);
      const output = await batcher.call(input);
      const outputBuffers = output.map(
        typed => typed.buffer
      );
      thread.output(Transfer(outputBuffers, outputBuffers));
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
      const gameHistory = await worker.play();
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
