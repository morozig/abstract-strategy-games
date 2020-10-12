import {
  Subject
} from 'threads/observable';
import GameRules from '../interfaces/game-rules';
import GameHistory from '../interfaces/game-history';
import PolicyAgent from '../interfaces/policy-agent';
import PolicyAction from '../interfaces/policy-action';
import GameModel from '../interfaces/game-model';

export default class PlayWorker {
  private gameRules: GameRules;
  private 
  constructor(gameRules: GameRules) {
    this.gameRules = gameRules;
  }
};
