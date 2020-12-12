import { expose } from 'threads/worker';
import createPlayWorker from '../../lib/play-worker';
import Rules from './rules';

const rules = new Rules(5, 5, 4);
const worker = createPlayWorker(rules);

expose(worker);
