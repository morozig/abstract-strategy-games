import * as tf from '@tensorflow/tfjs';

const convLayer2D = (
    input: tf.SymbolicTensor,
    options: {
        numFilters: number;
        kernelSize?: number;
    }
) => {
    const kernelSize = options.kernelSize || 3;
    let network = tf.layers.conv2d({
        kernelSize,
        filters: options.numFilters,
        strides: 1,
        padding: 'same',
    }).apply(input) as tf.SymbolicTensor;
    network = tf.layers.batchNormalization()
        .apply(network) as tf.SymbolicTensor;
    network = tf.layers.leakyReLU()
        .apply(network) as tf.SymbolicTensor;
    return network;
};

const residualLayer2D = (
    input: tf.SymbolicTensor,
    options: {
        numFilters: number;
        kernelSize?: number;
    }
) => {
    const kernelSize = options.kernelSize || 3;
    let network = convLayer2D(input, options);

    network = tf.layers.conv2d({
        kernelSize,
        filters: options.numFilters,
        strides: 1,
        padding: 'same',
    }).apply(input) as tf.SymbolicTensor;
    network = tf.layers.batchNormalization()
        .apply(network) as tf.SymbolicTensor;

    network = tf.layers.add()
        .apply([network, input]) as tf.SymbolicTensor;;
    network = tf.layers.leakyReLU()
        .apply(network) as tf.SymbolicTensor;;
    return network;
};

const residualNetwork2D = (
    input: tf.SymbolicTensor,
    options: {
        numLayers: number;
        numFilters: number;
        kernelSize?: number;
    }
) => {
    let network = convLayer2D(input, options);

    for (let i = 0; i < options.numLayers; i++) {
        network = residualLayer2D(network, options)
    }
    return network;
};

export {
    residualNetwork2D
}