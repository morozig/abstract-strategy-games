import GameModel, { TrainOptions } from '../../interfaces/game-model';
import GameHistory from '../../interfaces/game-history';
import State from './state';
import Tile from './tile';
import Rules from './rules';
import Network from './network';
import Batcher from '../../lib/batcher';
import config from '../../config';
import PolicyAction from '../../interfaces/policy-action';

// import { indexMax } from '../../lib/helpers';

type Input = number[][][];
type Output = [number[], number];

interface Pair {
    input: Input;
    output: Output;
};

const historyDepth = 2;
const useColor = true;

const getHash = (state: State) => {
    return state.board
        .map(row => row.join(''))
        .join('');
};

const getInput = (states: State[]) => {
    const lastState = states[states.length - 1];
    const playerIndex = lastState.playerIndex;
    const enemyIndex = 1 - playerIndex;
    const playerTile = playerIndex === 0 ? Tile.X : Tile.O;
    const enemyTile = playerIndex === 1 ? Tile.X : Tile.O;

    const input = [] as Input;

    for (let i = 0; i < 6; i++) {
        input.push([]);
        for (let j = 0; j < 7; j++) {
            input[i].push([]);
            const color = 1 - playerIndex;
            const playerHistory = states
                .filter(state => state.playerIndex === enemyIndex)
                .map(state => (
                    state.board[i][j] === playerTile ? 1 : 0
                ))
                .reverse()
                .slice(0, historyDepth) as number[];
            const enemyHistory = states
                .filter(state => state.playerIndex === playerIndex)
                .map(state => (
                    state.board[i][j] === enemyTile ? 1 : 0
                ))
                .reverse()
                .slice(0, historyDepth) as number[];
            const emptyPlayerHistory = [] as number[];
            const emptyEnemyHistory = [] as number[];
            const emptyPlayerHistoryLength = Math.max(
                historyDepth - playerHistory.length, 0
            );
            const emptyEnemyHistoryLength = Math.max(
                historyDepth - enemyHistory.length, 0
            );
            if (emptyPlayerHistoryLength) {
                for (let n = 0; n < emptyPlayerHistoryLength; n++) {
                    emptyPlayerHistory.push(0);
                }
            }
            if (emptyEnemyHistoryLength) {
                for (let n = 0; n < emptyEnemyHistoryLength; n++) {
                    emptyEnemyHistory.push(0);
                }
            }
            input[i][j] = playerHistory.concat(
                emptyPlayerHistory,
                enemyHistory,
                emptyEnemyHistory,
                useColor ? [color] : []
            );
        }
    }
    const hash = getHash(lastState);

    return {
        input,
        hash
    };
};

const getSymHistories = (history: PolicyAction[]) => {
    const symHistories = [history];
    symHistories.push(history.map(({ action, policy }) => ({
        action: 8 - action,
        policy: policy.slice().reverse()
    })));
    return symHistories;
};

const getStates = (history: number[], rules: Rules) => {
    const initial = rules.init();
    const states = [initial];
    let last = initial;
    for (let action of history) {
        const { state } = rules.step(last, action);
        states.push(state);
        last = state;
    }
    return states;
};

const getOutput = (reward: number, policy: number[]) => {
    const policyOutput = policy;
    const rewardOutput = reward;
    return [policyOutput, rewardOutput] as Output;
};

const networkPredictor = (network: Network) => {
    return (inputs: Input[]) => {
        return network.predict(inputs);
    };
};

export default class Model implements GameModel {
    private rules: Rules;
    private network: Network;
    private gameName: string;
    private batcher: Batcher<Input,Output> | null = null;
    constructor(gameName: string, rules: Rules, parallel = false) {
        this.gameName = gameName;
        this.rules = rules;
        this.network = new Network({
            historyDepth,
            useColor
        });
        if (parallel) {
            this.batcher = new Batcher(
                networkPredictor(this.network),
                100,
                10
            );
        }
    }
    async train(
        gameHistories: GameHistory[],
        options?: TrainOptions
    ) {
        const inputs = [] as Input[];
        const outputs = [] as Output[];
        const pairs = [] as Pair[];
        for (let gameHistory of gameHistories) {
            const symHistories = getSymHistories(gameHistory.history);
            for (let policyActions of symHistories) {
                const history = policyActions.map(({action}) => action);
                const states = getStates(history, this.rules);
                states.pop();
                for (let i = 0; i < states.length; i++) {
                    const layerStates = states.slice(0, i + 1);
                    const { input } = getInput(layerStates);
                    const lastState = states[i];
                    const lastPlayerIndex = lastState.playerIndex;
                    const reward = gameHistory.rewards[lastPlayerIndex];
                    const { policy } = gameHistory.history[i];
                    const output = getOutput(reward, policy);
                    pairs.push({
                        input,
                        output
                    });
                }
            }
        }
        pairs.forEach(pair => {
            inputs.push(pair.input);
            outputs.push(pair.output)
        });
        console.log(`training data length: ${inputs.length}`);
        if (options && options.improve) {
            this.network.addLayer();
        }
        const loss = await this.network.fit(inputs, outputs);
        return loss < 0.5;
    }
    async save(name: string){
        const location = window.location;
        const url = `${location}api/${this.gameName}/model/${name}`;
        await this.network.save(url);
    }
    async load(name: string){
        const url = `${config.dataUrl}/${this.gameName}/model/` +
            `${name}/model.json`;
        await this.network.load(url);
    }
    async predict(history: number[]) {
        const states = getStates(history, this.rules);
        const { input } = getInput(states);
        let output: Output;
        if (!this.batcher) {
            [output] = await this.network.predict([input]);
        } else {
            output = await this.batcher.call(input);
        }
        const [ policy, reward ] = output;
        return {
            reward,
            policy
        };
    }
};