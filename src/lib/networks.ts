import * as tf from '@tensorflow/tfjs';

const convLayer2D = (
    input: tf.SymbolicTensor,
    options: {
        numFilters: number;
        kernelSize?: number;
        name: string;
        noActivation?: boolean;
    }
) => {
    const kernelSize = options.kernelSize || 3;
    const useActivation = !options.noActivation;
    let network = tf.layers.conv2d({
        kernelSize,
        filters: options.numFilters,
        strides: 1,
        padding: 'same',
        name: options.name,
        useBias: false
    }).apply(input) as tf.SymbolicTensor;
    network = tf.layers.batchNormalization({
        name: `${options.name}_bn`,
        axis: 3
    })
        .apply(network) as tf.SymbolicTensor;
    if (useActivation) {
        network = tf.layers.leakyReLU()
            .apply(network) as tf.SymbolicTensor;
    }
    return network;
};

const residualLayer2D = (
    input: tf.SymbolicTensor,
    options: {
        id: number;
        numFilters: number;
        kernelSize?: number;
    }
) => {
    let network = convLayer2D(input, {
        name: `residual${options.id}_conv2d1`,
        numFilters: options.numFilters,
        kernelSize: options.kernelSize
    });
    network = convLayer2D(network, {
        name: `residual${options.id}_conv2d2`,
        numFilters: options.numFilters,
        kernelSize: options.kernelSize,
        noActivation: true
    });

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
    let network = convLayer2D(input, {
        name: 'residual0_conv2d',
        numFilters: options.numFilters,
        kernelSize: options.kernelSize
    });

    for (let i = 1; i <= options.numLayers; i++) {
        network = residualLayer2D(network, {
            id: i,
            numFilters: options.numFilters,
            kernelSize: options.kernelSize
        })
    }
    return network;
};

const copyWeights = (from: tf.LayersModel, to: tf.LayersModel) => {
    for (let sourceLayer of from.layers) {
        const sourceWeights = sourceLayer.getWeights();
        if (!sourceWeights || sourceWeights.length <= 0) {
            continue;
        }
        try {
            const targetLayer = to.getLayer(sourceLayer.name);
            targetLayer.setWeights(sourceWeights);
        }
        catch (err) {
            console.log(`Failed to copy layer ${sourceLayer.name}: ${err}`, from.layers);
        }
    }
};

const countResidualLayers = (model: tf.LayersModel) => {
    const totalResidualConvLayers = model
        .layers
        .filter(layer => layer.name.startsWith('residual'))
        .length;
    return (totalResidualConvLayers - 2) / 4;
};

export {
    convLayer2D,
    residualNetwork2D,
    copyWeights,
    countResidualLayers
}