import GameHistory from '../interfaces/game-history';
import * as tf from '@tensorflow/tfjs';
import config from '../config';
import { TrainExamples } from '../interfaces/game-rules';

const getHistories = async (gameName: string) => {
  const requestUrl = `/api/${gameName}/history`;
  const historiesRequest = await fetch(requestUrl);
  const histories = await historiesRequest.json() as string[];
  return histories;
};

const getModels = async (gameName: string) => {
  const requestUrl = `/api/${gameName}/model`;
  const modelsRequest = await fetch(requestUrl);
  const models = await modelsRequest.json() as string[];
  return models;
};

const loadHistory = async(gameName: string, historyName: string) => {
  const requestUrl = `/data/${gameName}/history/${historyName}.json`;
  const historyRequest = await fetch(requestUrl);
  const gameHistories = await historyRequest.json() as GameHistory[];
  return gameHistories;
};

const saveHistory = async(
    gameName: string,
    historyName: string,
    gameHistories: GameHistory[]
  ) => {
  const requestUrl = `/api/${gameName}/history`;
  const saveData = new FormData();
  saveData.append(
    `${historyName}.json`,
    new Blob(
      [JSON.stringify(gameHistories)],
      {type: 'application/json'}
    ),
    `${historyName}.json`
  );

  await fetch(
    requestUrl,
    {
      method: 'POST',
      body: saveData
    }
  );
};

const loadModel = async(gameName: string, modelName: string) => {
  const requestUrl = `${
    config.dataUrl
  }/${gameName}/model/${modelName}/model.json`;
  const model = await tf.loadLayersModel(requestUrl);
  return model;
};

const saveModel = async(
  model: tf.LayersModel,
  gameName: string,
  modelName: string
) => {
  const requestUrl = `${
    window.location
  }api/${gameName}/model/${modelName}`;
  await model.save(requestUrl);
};

const getTrainDir = async (
  gameName: string,
  modelName: string
) => {
  const requestUrl = `/api/${gameName}/train/${modelName}`;
  const trainDirRequest = await fetch(requestUrl);
  const trainDir = await trainDirRequest.json() as string[];
  return trainDir;
};

const loadTrainExamples = async(
  gameName: string,
  modelName: string
) => {
  const requestUrl = `/data/${gameName}/train/${modelName}/examples.json`;
  const trainExamplesRequest = await fetch(requestUrl);
  const trainExamples = await trainExamplesRequest.json() as TrainExamples;
  return trainExamples;
};

const saveTrainExamples = async(
    gameName: string,
    modelName: string,
    trainExamples: TrainExamples
  ) => {
  const requestUrl = `/api/${gameName}/train/${modelName}`;
  const saveData = new FormData();
  saveData.append(
    'examples.json',
    new Blob(
      [JSON.stringify(trainExamples)],
      {type: 'application/json'}
    ),
    'examples.json'
  );

  await fetch(
    requestUrl,
    {
      method: 'POST',
      body: saveData
    }
  );
};

const loadTrainLosses = async(
  gameName: string,
  modelName: string
) => {
  const requestUrl = `/data/${gameName}/train/${modelName}/losses.json`;
  const lossesRequest = await fetch(requestUrl);
  const losses = await lossesRequest.json() as number[][];
  return losses;
};

const saveTrainLosses = async(
    gameName: string,
    modelName: string,
    trainLosses: number[][]
  ) => {
  const requestUrl = `/api/${gameName}/train/${modelName}`;
  const saveData = new FormData();
  saveData.append(
    'losses.json',
    new Blob(
      [JSON.stringify(trainLosses)],
      {type: 'application/json'}
    ),
    'losses.json'
  );

  await fetch(
    requestUrl,
    {
      method: 'POST',
      body: saveData
    }
  );
};

const loadTrainModel = async(
  gameName: string,
  modelName: string
) => {
  const requestUrl = `${
    config.dataUrl
  }/${gameName}/train/${modelName}/model/model.json`;
  const model = await tf.loadLayersModel(requestUrl);
  return model;
};

const saveTrainModel = async(
  model: tf.LayersModel,
  gameName: string,
  modelName: string
) => {
  const requestUrl = `${
    window.location
  }api/${gameName}/train/${modelName}/model`;
  await model.save(requestUrl);
};

const deleteTrain = async(
  gameName: string
) => {
  const requestUrl = `/api/${gameName}/train`;

  await fetch(
    requestUrl,
    {
      method: 'DELETE'
    }
  );
};

export {
  getHistories,
  getModels,
  saveHistory,
  loadHistory,
  saveModel,
  loadModel,
  getTrainDir,
  loadTrainExamples,
  saveTrainExamples,
  loadTrainLosses,
  saveTrainLosses,
  loadTrainModel,
  saveTrainModel,
  deleteTrain
}