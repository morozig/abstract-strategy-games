import GameHistory from '../interfaces/game-history';
import Batcher from '../lib/batcher';
import PolicyAction from '../interfaces/policy-action';
import { PlaneSymmetry, plane } from '../lib/transforms';
import { softMax } from '../lib/helpers';

type Input = number[][][];
type Output = [number[], number];

interface Pair {
  input: Input;
  output: Output;
};

const networkPredictor = (network: Network) => {
    return (inputs: Input[]) => {
        return network.predict(inputs);
    };
};

export default class AlphaModel {
    private rules: Rules;
    private network: Network;
    private gameName: string;
    constructor(gameName: string, rules: Rules) {
        this.gameName = gameName;
        this.rules = rules;
        this.network = new Network({
            height: rules.height,
            width: rules.width,
            depth: this.depth
        });
    }
    async train(
        gameHistories: GameHistory[],
        options?: TrainOptions
    ) {
        const inputs = [] as Input[];
        const outputs = [] as Output[];
        const pairs = [] as Pair[];
        for (let gameHistory of gameHistories) {
            const symHistories = getSymHistories(
                gameHistory.history,
                this.rules
            );
            for (let policyActions of symHistories) {
                const history = policyActions.map(({action}) => action);
                const states = getStates(history, this.rules);
                states.pop();
                for (let i = 0; i < states.length; i++) {
                    const layerStates = states.slice(0, i + 1);
                    const { input } = getInput(layerStates, this.rules);
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
        await this.network.save(this.gameName, name);
    }
    async load(name: string){
        await this.network.load(this.gameName, name);
    }
    async predict(history: number[]) {
        const states = getStates(history, this.rules);
        const { input } = getInput(states, this.rules);
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
