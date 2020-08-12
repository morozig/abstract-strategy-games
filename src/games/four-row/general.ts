import * as tf from '@tensorflow/tfjs';
import { countResidualLayers, copyWeights } from '../../lib/networks';
import {
    saveModel,
    loadModel
} from '../../lib/api';

const numFilters = 512;
const defaultNumLayers = 4;
const numEpochs = 10;
const dropout = 0.3;

interface Options {
    historyDepth: number;
    useColor: boolean;
}

export default class Network {
    private model: tf.LayersModel;
    private historyDepth: number;
    private useColor: boolean;
    constructor(options: Options) {
        this.historyDepth = options.historyDepth;
        this.useColor = options.useColor;
        this.model = this.createModel();
        this.compile();
    }
    private createModel(numLayers=defaultNumLayers){
        const colorDepth = this.useColor ? 1 : 0;
        const depth = this.historyDepth * 2 + colorDepth;
        const input = tf.input({
            shape: [6, 7, depth]
        });

        let network = tf.layers.conv2d({
            kernelSize: 3,
            filters: numFilters,
            strides: 1,
            padding: 'same',
            useBias: false
        }).apply(input) as tf.SymbolicTensor;
        network = tf.layers.batchNormalization({
            axis: 3
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.activation({
            activation: 'relu'
        }).apply(network) as tf.SymbolicTensor;

        network = tf.layers.conv2d({
            kernelSize: 3,
            filters: numFilters,
            strides: 1,
            padding: 'same',
            useBias: false
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.batchNormalization({
            axis: 3
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.activation({
            activation: 'relu'
        }).apply(network) as tf.SymbolicTensor;

        network = tf.layers.conv2d({
            kernelSize: 3,
            filters: numFilters,
            strides: 1,
            padding: 'valid',
            useBias: false
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.batchNormalization({
            axis: 3
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.activation({
            activation: 'relu'
        }).apply(network) as tf.SymbolicTensor;

        network = tf.layers.conv2d({
            kernelSize: 3,
            filters: numFilters,
            strides: 1,
            padding: 'valid',
            useBias: false
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.batchNormalization({
            axis: 3
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.activation({
            activation: 'relu'
        }).apply(network) as tf.SymbolicTensor;

        network = tf.layers.flatten(
        ).apply(network) as tf.SymbolicTensor;

        network = tf.layers.dense({
            units: 1024
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.batchNormalization({
            axis: 1
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.activation({
            activation: 'relu'
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.dropout({
            rate: dropout
        }).apply(network) as tf.SymbolicTensor;

        network = tf.layers.dense({
            units: 512
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.batchNormalization({
            axis: 1
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.activation({
            activation: 'relu'
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.dropout({
            rate: dropout
        }).apply(network) as tf.SymbolicTensor;



        let policy = tf.layers.dense({
            units: 7
        }).apply(network) as tf.SymbolicTensor;
        policy = tf.layers.softmax(
        ).apply(policy) as tf.SymbolicTensor;

        let reward = tf.layers.dense({
            units: 1
        }).apply(network) as tf.SymbolicTensor;
        reward = tf.layers.activation({
            activation: 'tanh'
        }).apply(reward) as tf.SymbolicTensor;

        const model = tf.model(
            {
                inputs: input,
                outputs: [
                    policy,
                    reward
                ]
            }
        );
        return model;
    };
    private compile() {
        const optimizer = tf.train.adam();
        // const optimizer = tf.train.sgd(0.1);

        this.model.compile({
            optimizer: optimizer,
            loss: [
                'categoricalCrossentropy',
                'meanSquaredError'
            ],
            metrics: ['accuracy']
        });
    }
    async fit(
        inputs: number[][][][],
        outputs: [number[], number][]
    ){
        // const batchSize = inputs.length;
        const xsTensor = tf.tensor4d(inputs);
        const policiesTensor = tf.tensor2d(outputs.map(
            output => output[0]
        ));
        const rewardsTensor = tf.tensor2d(outputs.map(
            output => [output[1]]
        ));
        const ysTensors = [
            policiesTensor,
            rewardsTensor
        ];

        const trainingHistory = await this.model.fit(
            xsTensor,
            ysTensors,
            {
                batchSize: 128,
                epochs: numEpochs,
                shuffle: true,
                validationSplit: 0.01,
                callbacks: {
                    onEpochEnd: console.log
                }
            }
        );

        xsTensor.dispose();
        policiesTensor.dispose();
        rewardsTensor.dispose();
        console.log(trainingHistory);
        const loss = trainingHistory.history.loss[numEpochs - 1] as number;
        return loss;
    }
    async predict(inputs: number[][][][]) {
        const inputsTensor = tf.tensor4d(inputs);
        const [ policiesTensor, rewardsTensor ] = this.model.predict(
            inputsTensor
        ) as [tf.Tensor2D, tf.Tensor2D];
        const policies = await policiesTensor.array();
        const rewards = await rewardsTensor.array();

        inputsTensor.dispose();
        policiesTensor.dispose();
        rewardsTensor.dispose();

        const outputs = policies.map(
            (policy, i) => [policy, rewards[i][0]] as [number[], number]
        );
        return outputs;
    }
    async save(gameName: string, modelName: string) {
        await saveModel(
            this.model,
            gameName,
            modelName
        );
    }
    async load(gameName: string, modelName: string) {
        this.model.dispose();
        this.model = await loadModel(gameName, modelName);
        this.compile();
    }
    addLayer() {
        const numLayers = countResidualLayers(this.model);
        console.log(`new layer: ${numLayers + 1}`);
        const newModel = this.createModel(numLayers + 1);
        copyWeights(this.model, newModel);
        this.model.dispose();
        this.model = newModel;
        this.compile();
    }
};