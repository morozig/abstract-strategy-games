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
    }).apply(network) as tf.SymbolicTensor;
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

const copyWeights = (from: tf.LayersModel, to: tf.LayersModel) => {
    for (let sourceLayer of from.layers) {
        try {
            const targetLayer = to.getLayer(sourceLayer.name);
            const sourceWeights = sourceLayer.getWeights();
            if (sourceWeights.length > 0) {
                targetLayer.setWeights(sourceWeights);
            }
        }
        catch (err) {
            console.log(`Failed to copy layer ${sourceLayer.name}: ${err}`);
        }
    }
};

const isResidualEnd = (output: tf.layers.Layer) => {
    const name = output.name;
    if (!name.startsWith('leaky_re_lu_')) {
        return false;
    }
    const addLayer = output.input;
    if (!addLayer) {
        return false;
    }
    if (!(addLayer instanceof tf.SymbolicTensor)) {
        return false;
    }
    if (!addLayer.name.startsWith('add_')) {
        return false;
    }
    const [ batch2Layer , input2 ] = addLayer.inputs;
    if (!batch2Layer || !input2) {
        return false;
    }
    if (!batch2Layer.name.startsWith('batch_normalization_')) {
        return false;
    }
    const [ conv2Layer ] = batch2Layer.inputs;
    if (!conv2Layer) {
        return false;
    }
    if (!conv2Layer.name.startsWith('conv2d_')) {
        return false;
    }
    const [ leakyLayer ] = conv2Layer.inputs;
    if (!leakyLayer) {
        return false;
    }
    if (!leakyLayer.name.startsWith('leaky_re_lu_')) {
        return false;
    }
    const [ batch1Layer ] = leakyLayer.inputs;
    if (!batch1Layer) {
        return false;
    }
    if (!batch1Layer.name.startsWith('batch_normalization_')) {
        return false;
    }
    const [ conv1Layer ] = batch1Layer.inputs;
    if (!conv1Layer) {
        return false;
    }
    if (!conv1Layer.name.startsWith('conv2d_')) {
        return false;
    }
    const [ input1 ] = conv1Layer.inputs;
    if (!input1) {
        return false;
    }
    if (!(input1.name === input2.name)) {
        return false;
    }
    return true;
};

const countResidualLayers = (model: tf.LayersModel) => {
    return model
        .layers
        .filter(isResidualEnd)
        .length;
};

export {
    residualNetwork2D,
    copyWeights,
    countResidualLayers
}