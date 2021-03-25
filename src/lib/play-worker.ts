import {
  Observable,
  Subject
} from 'threads/observable';
import GameRules from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import { Transfer, TransferDescriptor } from 'threads';
import { play } from '../lib/play';
import Mcts from '../agents/mcts';
import GamePrediction from '../interfaces/game-prediction';
import Batcher from './batcher';
import { sleep } from './helpers';

interface InputsRequest {
  inputs: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>;
  modelIndex?: number;
};

interface OutputsResponse {
  policies: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>;
  rewards: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>;
  modelIndex?: number;
};

export interface PlayWorkerOptions {
  gameName?: string;
  modelsIndices?: number[];
  planCount: number;
  randomTurnsCount?: number;
};

export type PlayWorkerType = {
  play: (options: PlayWorkerOptions) => Promise<GameHistory>;
  inputs: () => Observable<InputsRequest>;
  setOutputs: (outputsResponse: OutputsResponse) => void;
};

type PredictBatchResolver = (
  [ policyBuffers, rewardBuffers ]:
  [ ArrayBuffer[], ArrayBuffer[] ]
) => void;

const createPlayWorker = (
  gameRules: GameRules
) => {
  const inputsSubject = new Subject<InputsRequest>();
  const queue = [] as PredictBatchResolver[];
  const modelsQueue = new Array(gameRules.playersCount)
    .fill(undefined)
    .map(_ => [] as PredictBatchResolver[]);
  const predictBatch = async (
    histories: number[][],
    modelIndex?: number
  ) => {
    const inputBuffers = histories
      .map(history => gameRules.getInput(history))
      .map(input => new Float32Array(input.flat(2)))
      .map(typedInput => typedInput.buffer);
    inputsSubject.next({
      inputs: Transfer(inputBuffers, inputBuffers),
      modelIndex
    });
    const [ policyBuffers, rewardBuffers ] = await new Promise(
      (resolve: PredictBatchResolver) => {
        if (modelIndex === undefined) {
          queue.push(resolve);
        } else {
          modelsQueue[modelIndex].push(resolve);
        }
      }
    );
    const predictions = policyBuffers.map(
      (policyBuffer, i) => {
        const rewardBuffer = rewardBuffers[i];
        const policy = [] as number[];
        const rewards = [] as number[];
        const policyTyped = new Float32Array(policyBuffer);
        policyTyped.forEach(prob => policy.push(prob));
        const rewardsTyped = new Float32Array(rewardBuffer);
        rewardsTyped.forEach(value => rewards.push(value));
        return {
          policy,
          reward: rewards[0]
        } as GamePrediction
      }
    );
    return predictions;
  };
  const batcher = new Batcher(
    (histories: number[][]) => predictBatch(histories),
    0,
    10
  );
  const modelsBatchers = new Array(gameRules.playersCount)
    .fill(undefined)
    .map((_, i) => new Batcher(
      (histories: number[][]) => predictBatch(histories, i),
      0,
      10
    ));
  const predict = (
    history: number[],
    modelIndex?: number
  ) => {
    if (modelIndex === undefined) {
      return batcher.call(history);
    } else {
      return modelsBatchers[modelIndex].call(history);
    }
  }

  return {
    inputs() {
      return Observable.from(inputsSubject)
    },
    setOutputs(outputsResponse: OutputsResponse) {
      const resolve = outputsResponse.modelIndex === undefined ?
        queue.shift() :
        modelsQueue[outputsResponse.modelIndex].shift();
      if (resolve) {
        resolve([
          ('send' in outputsResponse.policies) ?
            outputsResponse.policies.send :
            outputsResponse.policies,
          ('send' in outputsResponse.rewards) ?
            outputsResponse.rewards.send :
            outputsResponse.rewards
        ]);
      }
    },
    async play(options: PlayWorkerOptions) {
      if (options.modelsIndices) {
        for (let batcher of modelsBatchers) {
          batcher.setSize(size => size + 1);
        }
      } else {
        batcher.setSize(size => size + 1);
      }
      await sleep(10);
      const gameHistory = await play(
        gameRules,
        [
          new Mcts({
            gameRules: gameRules,
            planCount: options.planCount,
            predict: (history: number[]) => predict(
              history,
              options.modelsIndices && options.modelsIndices[0]
            ),
            randomize: true,
            randomTurnsCount: options.randomTurnsCount
          }),
          new Mcts({
            gameRules: gameRules,
            planCount: options.planCount,
            predict: (history: number[]) => predict(
              history,
              options.modelsIndices && options.modelsIndices[1]
            ),
            randomize: true,
            randomTurnsCount: options.randomTurnsCount
          }),
        ],
        options.gameName
      );
      if (options.modelsIndices) {
        for (let batcher of modelsBatchers) {
          batcher.setSize(size => size - 1);
        }
      } else {
        batcher.setSize(size => size - 1);
      }
      return gameHistory;
    }
  } as PlayWorkerType;
};

export default createPlayWorker;
