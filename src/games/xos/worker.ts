import { expose } from 'threads/worker';
import createPlayWorker from '../../lib/play-worker';
import config from './config.json';
import Rules from './rules';

const rules = new Rules(config.height, config.width, config.same);
const worker = createPlayWorker(rules);

expose(worker);
