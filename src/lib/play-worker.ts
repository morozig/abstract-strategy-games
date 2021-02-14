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

export type PlayWorkerType = {
  play: (gameName?: string) => Promise<GameHistory>;
  inputs: () => Observable<
    ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>
  >;
  setOutputs: (
    policies: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>,
    rewards: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>
  ) => void;
};

type PredictBatchResolver = (
  [ policyBuffers, rewardBuffers ]:
  [ ArrayBuffer[], ArrayBuffer[] ]
) => void;

const createPlayWorker = (
  gameRules: GameRules,
  planCount = 800
) => {
  const inputsSubject = new Subject<
    ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>
  >();
  const queue = [] as PredictBatchResolver[];
  const predictBatch = async (histories: number[][]) => {
    const inputBuffers = histories
      .map(history => gameRules.getInput(history))
      .map(input => new Float32Array(input.flat(2)))
      .map(typedInput => typedInput.buffer);
    inputsSubject.next(Transfer(inputBuffers, inputBuffers));
    const [ policyBuffers, rewardBuffers ] = await new Promise(
      (resolve: PredictBatchResolver) => {
        queue.push(resolve);
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
  const predict = (history: number[]) => batcher.call(history);

  return {
    inputs() {
      return Observable.from(inputsSubject)
    },
    setOutputs(
      policyBuffers: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>,
      rewardBuffers: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>
    ) {
      const resolve = queue.shift();
      if (resolve) {
        resolve([
          ('send' in policyBuffers) ?
            policyBuffers.send :
            policyBuffers,
          ('send' in rewardBuffers) ?
            rewardBuffers.send :
            rewardBuffers
        ]);
      }
    },
    async play(gameName?: string) {
      batcher.setSize(size => size + 1);
      await sleep(10);
      const gameHistory = await play(
        gameRules,
        [
          new Mcts({
            gameRules: gameRules,
            planCount: planCount,
            predict: (history: number[]) => predict(history),
            randomize: true
          }),
          new Mcts({
            gameRules: gameRules,
            planCount: planCount,
            predict: (history: number[]) => predict(history),
            randomize: true
          }),
        ],
        gameName
      );
      batcher.setSize(size => size - 1);
      return gameHistory;
    }
  } as PlayWorkerType;
};

export default createPlayWorker;
