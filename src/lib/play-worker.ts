import {
  Observable,
  Subject
} from 'threads/observable';
import GameRules, { Input, Output } from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import PolicyAgent from '../interfaces/policy-agent';
import PolicyAction from '../interfaces/policy-action';

export type PlayWorkerType = {
  play: () => Promise<GameHistory>;
  inputs: () => Observable<Input[]>;
  outputs: () => Subject<Output[]>;
};
export default class PlayWorker implements PlayWorkerType
{
  private gameRules: GameRules;
  constructor(gameRules: GameRules) {
    this.gameRules = gameRules;
  }
  
};
