import PolicyAction from './policy-action';

export default interface GameHistory {
  readonly history: PolicyAction[];
  readonly rewards: number[];
};
