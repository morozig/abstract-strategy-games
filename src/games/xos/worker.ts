import { expose } from 'threads/worker';
import PlayWorker, { PlayWorkerType } from '../../lib/play-worker';
import Game from './index';

const game = new Game();
const worker = new PlayWorker(game.rules);

expose(worker as PlayWorkerType);
