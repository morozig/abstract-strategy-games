import * as tf from '@tensorflow/tfjs';
import {
    saveModel,
    loadModel
} from '../../../lib/api';

const numFilters = 32;
const defaultNumLayers = 4;
const numEpochs = 10;

interface Options {
    height: number;
    width: number;
    depth: number;
}

export default class Network {
    private model: tf.LayersModel;
    private height: number;
    private width: number;
    private depth: number;
    constructor(options: Options) {
        this.height = options.height;
        this.width = options.width;
        this.depth = options.depth;
        this.model = this.createModel();
        this.compile();
    }
    private createModel(numLayers=defaultNumLayers){
        const input = tf.input({
            shape: [this.height, this.width, this.depth]
        });

        let network = tf.layers.conv2d({
            kernelSize: 2,
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
            kernelSize: 2,
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


        let policy = tf.layers.conv2d({
            kernelSize: 1,
            filters: 1,
            strides: 1,
            padding: 'same',
            useBias: false
        }).apply(network) as tf.SymbolicTensor;
        policy = tf.layers.batchNormalization({
            axis: 3
        }).apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.activation({
            activation: 'relu'
        }).apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.flatten(
        ).apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.dense({
            units: this.height * this.width
        }).apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.softmax(
        ).apply(policy) as tf.SymbolicTensor;

        let reward = tf.layers.conv2d({
            kernelSize: 2,
            filters: 1,
            strides: 1,
            padding: 'valid',
            useBias: false
        }).apply(network) as tf.SymbolicTensor;
        reward = tf.layers.batchNormalization({
            axis: 3
        }).apply(reward) as tf.SymbolicTensor;
        reward = tf.layers.activation({
            activation: 'relu'
        }).apply(reward) as tf.SymbolicTensor;
        reward = tf.layers.flatten(
        ).apply(reward) as tf.SymbolicTensor;
        reward = tf.layers.dense({
            units: 1
        }).apply(reward) as tf.SymbolicTensor;
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
        const optimizer = tf.train.adam(0.001);
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

    }
};