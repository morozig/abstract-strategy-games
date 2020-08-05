const tf = require('@tensorflow/tfjs-node-gpu');

const run = async () => {
  const model = tf.sequential({
    layers: [tf.layers.dense({units: 1, inputShape: [10]})]
  });
  model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
  for (let i = 1; i < 5 ; ++i) {
    const h = await model.fit(tf.ones([8, 10]), tf.ones([8, 1]), {
        batchSize: 4,
        epochs: 3
    });
    console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
  }
};

run();
