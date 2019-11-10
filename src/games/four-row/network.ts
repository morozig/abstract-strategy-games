import * as tf from '@tensorflow/tfjs';

const numFilters = 70;
const numLayers = 1;
const numEpochs = 40;

interface Options {
    historyDepth: number;
    useColor: boolean;
}

const residual = ( input: tf.SymbolicTensor ) => {
    let network = tf.layers.conv2d({
        kernelSize: 3,
        filters: numFilters,
        strides: 1,
        padding: 'same',
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(input) as tf.SymbolicTensor;

    network = tf.layers.batchNormalization()
        .apply(network) as tf.SymbolicTensor;

    network = tf.layers.leakyReLU()
        .apply(network) as tf.SymbolicTensor;

    network = tf.layers.conv2d({
        kernelSize: 3,
        filters: numFilters,
        strides: 1,
        padding: 'same',
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(input) as tf.SymbolicTensor;

    network = tf.layers.batchNormalization()
        .apply(network) as tf.SymbolicTensor;

    network = tf.layers.add()
        .apply([network, input]) as tf.SymbolicTensor;;

    network = tf.layers.leakyReLU()
        .apply(network) as tf.SymbolicTensor;;


    return network;
};

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
    private createModel(){
        const colorDepth = this.useColor ? 1 : 0;
        const depth = this.historyDepth * 2 + colorDepth;
        console.log(depth);
        const input = tf.input({
            shape: [6, 7, depth]
        });

        let network = input;

        network = tf.layers.conv2d({
            kernelSize: 3,
            filters: numFilters,
            padding: 'same',
            strides: 1,
        }).apply(network) as tf.SymbolicTensor;
        network = tf.layers.batchNormalization()
            .apply(network) as tf.SymbolicTensor;
        network = tf.layers.leakyReLU()
            .apply(network) as tf.SymbolicTensor;

        for (let i = 0; i < numLayers; i++) {
            network = residual(network)
        }

        let policy = tf.layers.conv2d({
            kernelSize: 1,
            filters: 1,
            strides: 1,
            padding: 'same'
        }).apply(network) as tf.SymbolicTensor;
        policy = tf.layers.batchNormalization()
            .apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.leakyReLU()
            .apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.maxPooling2d(
            {
                poolSize: [6, 1]
            }
        ).apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.batchNormalization()
            .apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.flatten()
            .apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.dense({
            units: 7
        }).apply(policy) as tf.SymbolicTensor;
        policy = tf.layers.softmax()
            .apply(policy) as tf.SymbolicTensor;

        let reward = tf.layers.conv2d({
            kernelSize: 1,
            filters: 1,
            strides: 1,
            padding: 'same',
        }).apply(network) as tf.SymbolicTensor;
        reward = tf.layers.batchNormalization()
            .apply(reward) as tf.SymbolicTensor;
        reward = tf.layers.leakyReLU()
            .apply(reward) as tf.SymbolicTensor;
        reward = tf.layers.globalAveragePooling2d({})
            .apply(reward) as tf.SymbolicTensor;
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
    async save(url: string) {
        await this.model.save(url);
    }
    async load(url: string) {
        this.model = await tf.loadLayersModel(url);
        this.compile();
    }
};