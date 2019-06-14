import * as tf from '@tensorflow/tfjs';

const NUM_FILTERS = 75;

const residual = ( input: tf.SymbolicTensor ) => {
    let network = tf.layers.conv2d({
        kernelSize: 3,
        filters: NUM_FILTERS,
        strides: 1,
        padding: 'same',
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(input);

    network = tf.layers.batchNormalization({}).apply(network);

    network = tf.layers.leakyReLU().apply(network);

    network = tf.layers.conv2d({
        kernelSize: 3,
        filters: NUM_FILTERS,
        strides: 1,
        padding: 'same',
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(input);

    network = tf.layers.batchNormalization({}).apply(network);

    network = tf.layers.add().apply([network, input] as tf.SymbolicTensor[]);

    network = tf.layers.leakyReLU().apply(network);


    return network;
};

const get = async (url?: any) => {
    const input = tf.input({
        shape: [6, 7, 3]
    });

    let network = tf.layers.conv2d({
        inputShape: [6, 7, 3],
        kernelSize: 3,
        filters: NUM_FILTERS,
        strides: 1,
        padding: 'same',
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(input);

    network = tf.layers.batchNormalization({}).apply(network);
        
    network = tf.layers.leakyReLU().apply(network);

    network = residual(network as tf.SymbolicTensor);
    network = residual(network as tf.SymbolicTensor);
    network = residual(network as tf.SymbolicTensor);
    network = residual(network as tf.SymbolicTensor);
    network = residual(network as tf.SymbolicTensor);

    let policy = tf.layers.conv2d({
        kernelSize: 1,
        filters: 2,
        strides: 1,
        padding: 'same',
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(network);

    policy = tf.layers.batchNormalization({}).apply(policy);

    policy = tf.layers.leakyReLU().apply(policy);

    policy = tf.layers.flatten().apply(policy);

    policy = tf.layers.dense({
        units: 7,
        activation: 'softmax',
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(policy);


    let value = tf.layers.conv2d({
        kernelSize: 1,
        filters: 1,
        strides: 1,
        padding: 'same',
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(network);

    value = tf.layers.batchNormalization({}).apply(value);

    value = tf.layers.leakyReLU().apply(value);

    value = tf.layers.flatten().apply(value);

    value = tf.layers.dense({
        units: 20,
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(value);

    value = tf.layers.leakyReLU().apply(value);

    value = tf.layers.dense({
        units: 1,
        activation: 'tanh',
        // kernelInitializer: 'VarianceScaling',
        // kernelRegularizer: 'l1l2'
    }).apply(value);
    
    let model;
    if (!url) {
        model = tf.model(
            {
                inputs: input,
                outputs: [
                    policy as tf.SymbolicTensor,
                    value as tf.SymbolicTensor
                ]
            }
        );
    } else {
        model = await tf.loadModel(url);
    }

    // const LEARNING_RATE = 0.05;
    // const LEARNING_RATE = 0.0001;
    // const LEARNING_RATE = 0.05;
    // const optimizer = tf.train.adam(LEARNING_RATE);
    const optimizer = tf.train.adam();
    // const optimizer = tf.train.adadelta();
    // const optimizer = tf.train.sgd(LEARNING_RATE);

    model.compile({
        optimizer: optimizer,
        loss: [
            'categoricalCrossentropy',
            'meanSquaredError'
        ],
        metrics: ['accuracy']
    });
    return model;
}

export {
    get
}