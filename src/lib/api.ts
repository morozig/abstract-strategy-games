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

const getTrainDir = async (gameName: string) => {
  const requestUrl = `/api/${gameName}/train`;
  const trainDirRequest = await fetch(requestUrl);
  const trainDir = await trainDirRequest.json() as string[];
  return trainDir;
};

const loadTrainExamples = async(gameName: string) => {
  const requestUrl = `/data/${gameName}/train/examples.json`;
  const trainExamplesRequest = await fetch(requestUrl);
  const trainExamples = await trainExamplesRequest.json() as TrainExamples;
  return trainExamples;
};

const saveTrainExamples = async(
    gameName: string,
    trainExamples: TrainExamples
  ) => {
  const requestUrl = `/api/${gameName}/train`;
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

export {
  getHistories,
  getModels,
  saveHistory,
  loadHistory,
  saveModel,
  loadModel,
  getTrainDir,
  loadTrainExamples,
  saveTrainExamples
}