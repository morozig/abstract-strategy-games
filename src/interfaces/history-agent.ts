import Agent from './agent';
import HistoryAction from './history-action';

export default interface HistoryAgent extends Agent {
    historyAct(): Promise<HistoryAction>;
}