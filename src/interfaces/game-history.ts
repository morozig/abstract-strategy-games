import HistoryAction from './history-action';

export default interface GameHistory {
  readonly history: HistoryAction[];
  readonly rewards: number[];
};
