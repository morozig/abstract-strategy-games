import {
  Observable,
  Subject
} from 'threads/observable';
import GameRules, { Input, Output } from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import PolicyAgent from '../interfaces/policy-agent';
import PolicyAction from '../interfaces/policy-action';
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
export default class PlayWorker implements PlayWorkerType
{
  private gameRules: GameRules;
  private planCount: number;
  private inputsSubject = new Subject<
    ArrayBuffer | TransferDescriptor<ArrayBuffer>
  >();
  private queue = [] as PredictResolver[];
  constructor(gameRules: GameRules, planCount?: number) {
    this.gameRules = gameRules;
    this.planCount = planCount || 300;
  }
  private async predict(history: number[]) {
    const input = this.gameRules.getInput(history);
    const typedInput = new Float32Array(input.flat(2));
    this.inputsSubject.next(typedInput.buffer);
    const [ policyBuffer, valueBuffer ] = await new Promise(
      (resolve: PredictResolver) => {
        this.queue.push(resolve);
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

  inputs() {
    return Observable.from(this.inputsSubject)
  }
  output(
    outputBuffers: ArrayBuffer[] | TransferDescriptor<ArrayBuffer[]>
  ) {
    const resolve = this.queue.shift();
    if (resolve) {
      resolve(outputBuffers as ArrayBuffer[]);
    }
  }
  play() {
    const agent = new Mcts({
      gameRules: this.gameRules,
      planCount: this.planCount,
      predict: (history: number[]) => this.predict(history),
      randomize: true
    });
    return play(
      this.gameRules,
      [ agent, agent ]
    );
  }
};
