import { expose } from 'threads/worker';
import PlayWorker, { PlayWorkerType } from '../../lib/play-worker';
import Rules from './rules';

const rules = new Rules(5, 5, 4);
const worker = new PlayWorker(rules);

expose(worker as PlayWorkerType);
