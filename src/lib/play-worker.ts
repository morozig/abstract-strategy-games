import {
  Observable,
  Subject
} from 'threads/observable';
import GameRules from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import { TransferDescriptor } from 'threads';
import { play } from '../lib/play';
import Mcts from '../agents/mcts';

export type PlayWorkerType = {
  play: () => Promise<GameHistory>;
  inputs: () => Observable<ArrayBuffer | TransferDescriptor<ArrayBuffer>>;
  output: (
    outputBuffers: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>
  ) => void;
};

type PredictResolver = (outputBuffers: ArrayBuffer[]) => void;
const createPlayWorker = (gameRules: GameRules, planCount = 300) =>
{
  const inputsSubject = new Subject<
    ArrayBuffer | TransferDescriptor<ArrayBuffer>
  >();
  const queue = [] as PredictResolver[];
  const predict = async (history: number[]) => {
    const input = gameRules.getInput(history);
    const typedInput = new Float32Array(input.flat(2));
    inputsSubject.next(typedInput.buffer);
    const [ policyBuffer, valueBuffer ] = await new Promise(
      (resolve: PredictResolver) => {
        queue.push(resolve);
      }
    );
    const policy = [] as number[];
    const values = [] as number[];
    const policyTyped = new Float32Array(policyBuffer);
    policyTyped.forEach(prob => policy.push(prob));
    const valuesTyped = new Float32Array(valueBuffer);
    valuesTyped.forEach(value => values.push(value));

    return {
      policy,
      reward: values[0]
    }
  }

  return {
    inputs() {
      return Observable.from(inputsSubject)
    },
    output(
      outputBuffers: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>
    ) {
      const resolve = queue.shift();
      if (resolve) {
        resolve(outputBuffers as ArrayBuffer[]);
      }
    },
    play() {
      const agent = new Mcts({
        gameRules: gameRules,
        planCount: planCount,
        predict: (history: number[]) => predict(history),
        randomize: true
      });
      return play(
        gameRules,
        [ agent, agent ]
      );
    }
  } as PlayWorkerType;
};

export default createPlayWorker;
