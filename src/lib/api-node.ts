import GameHistory from '../interfaces/game-history';
import path from 'path';
import fs from 'fs';
import * as tf from '@tensorflow/tfjs';
import url from 'url';
import { TrainExamples } from '../interfaces/game-rules';

const dataDir = path.resolve(process.cwd(), 'data');

if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir);
}

const getHistories = async (gameName: string) => {
  const histories = [] as string[];
  const gameDir = path.resolve(dataDir, gameName);
  if (!fs.existsSync(gameDir)) {
    return histories;
  }
  const historyDir = path.resolve(gameDir, 'history');
  if (!fs.existsSync(historyDir)) {
    return histories;
  }

  for (let historyJson of fs.readdirSync(historyDir)) {
    histories.push(path.basename(historyJson, '.json'));
  }
  return histories;
};

const getModels = async (gameName: string) => {
  const models = [] as string[];
  const gameDir = path.resolve(dataDir, gameName);
  if (!fs.existsSync(gameDir)) {
    return models;
  }
  const modelsDir = path.resolve(gameDir, 'model');
  if (!fs.existsSync(modelsDir)) {
    return models;
  }
  for (let modelDir of fs.readdirSync(modelsDir)) {
    models.push(modelDir);
  }
  return models;
};

const loadHistory = async(gameName: string, historyName: string) => {
  const gameDir = path.resolve(dataDir, gameName);
  const historyDir = path.resolve(gameDir, 'history');
  const filePath = path.resolve(
    historyDir,
    `${historyName}.json`
  );

  const rawData = await fs.readFileSync(
    filePath, 'utf8'
  );
  const gameHistories = JSON.parse(
    rawData
  ) as GameHistory[];
  return gameHistories;
};

const saveHistory = async(
  gameName: string,
  historyName: string,
  gameHistories: GameHistory[]
) => {
  const gameDir = path.resolve(dataDir, gameName);
  if (!fs.existsSync(gameDir)) {
    fs.mkdirSync(gameDir);
  }
  const historyDir = path.resolve(gameDir, 'history');
  if (!fs.existsSync(historyDir)) {
    fs.mkdirSync(historyDir);
  }

  const filePath = path.resolve(
    historyDir,
    `${historyName}.json`
  );
  fs.writeFileSync(
    filePath,
    JSON.stringify(gameHistories),
    'utf8'
  );
};

const loadModel = async(gameName: string, modelName: string) => {
  const gameDir = path.resolve(dataDir, gameName);
  const modelsDir = path.resolve(gameDir, 'model');
  const modelDir = path.resolve(modelsDir, modelName);
  const modelPath = path.resolve(modelDir, 'model.json');
  const requestUrl = url.pathToFileURL(modelPath).href.replace(
    '///', 
    process.platform === 'win32' ? '//' : '///'
  ); // https://stackoverflow.com/questions/57859770

  const model = await tf.loadLayersModel(requestUrl);
  return model;
};

const saveModel = async(
  model: tf.LayersModel,
  gameName: string,
  modelName: string
) => {
  const gameDir = path.resolve(dataDir, gameName);
  if (!fs.existsSync(gameDir)) {
    fs.mkdirSync(gameDir);
  }
  const modelsDir = path.resolve(gameDir, 'model');
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir);
  }
  const modelDir = path.resolve(modelsDir, modelName);
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir);
  }

  const requestUrl = url.pathToFileURL(modelDir).href.replace(
    '///', 
    process.platform === 'win32' ? '//' : '///'
  ); // https://stackoverflow.com/questions/57859770
  await model.save(requestUrl);
};

const getTrainDir = async (
  gameName: string,
  modelName: string
) => {
  const files = [] as string[];
  const gameDir = path.resolve(dataDir, gameName);
  if (!fs.existsSync(gameDir)) {
    return files;
  }
  const trainDir = path.resolve(gameDir, 'train');
  if (!fs.existsSync(trainDir)) {
    return files;
  }
  const modelDir = path.resolve(trainDir, modelName);
  if (!fs.existsSync(modelDir)) {
    return files
  }

  for (let file of fs.readdirSync(modelDir)) {
    files.push(path.basename(file, '.json'));
  }
  return files;
};

const loadTrainExamples = async(
  gameName: string,
  modelName: string
) => {
  const gameDir = path.resolve(dataDir, gameName);
  const trainDir = path.resolve(gameDir, 'train');
  const modelDir = path.resolve(trainDir, modelName);

  const filePath = path.resolve(
    modelDir,
    'examples.json'
  );

  const rawData = await fs.readFileSync(
    filePath, 'utf8'
  );
  const trainExamples = JSON.parse(
    rawData
  ) as TrainExamples;
  return trainExamples;
};

const saveTrainExamples = async(
  gameName: string,
  modelName: string,
  trainExamples: TrainExamples
) => {
  const gameDir = path.resolve(dataDir, gameName);
  if (!fs.existsSync(gameDir)) {
    fs.mkdirSync(gameDir);
  }
  const trainDir = path.resolve(gameDir, 'train');
  if (!fs.existsSync(trainDir)) {
    fs.mkdirSync(trainDir);
  }
  const modelDir = path.resolve(trainDir, modelName);
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir);
  }

  const filePath = path.resolve(
    modelDir,
    'examples.json'
  );
  fs.writeFileSync(
    filePath,
    JSON.stringify(trainExamples),
    'utf8'
  );
};

const loadTrainLosses = async(
  gameName: string,
  modelName: string
) => {
  const gameDir = path.resolve(dataDir, gameName);
  const trainDir = path.resolve(gameDir, 'train');
  const modelDir = path.resolve(trainDir, modelName);
  const filePath = path.resolve(
    modelDir,
    'losses.json'
  );

  const rawData = await fs.readFileSync(
    filePath, 'utf8'
  );
  const trainLosses = JSON.parse(
    rawData
  ) as number[][];
  return trainLosses;
};

const saveTrainLosses = async(
  gameName: string,
  modelName: string,
  trainLosses: number[][]
) => {
  const gameDir = path.resolve(dataDir, gameName);
  if (!fs.existsSync(gameDir)) {
    fs.mkdirSync(gameDir);
  }
  const trainDir = path.resolve(gameDir, 'train');
  if (!fs.existsSync(trainDir)) {
    fs.mkdirSync(trainDir);
  }
  const modelDir = path.resolve(trainDir, modelName);
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir);
  }

  const filePath = path.resolve(
    modelDir,
    'losses.json'
  );
  fs.writeFileSync(
    filePath,
    JSON.stringify(trainLosses),
    'utf8'
  );
};

const loadTrainModel = async(
  gameName: string,
  modelName: string
) => {
  const gameDir = path.resolve(dataDir, gameName);
  const trainDir = path.resolve(gameDir, 'train');
  const modelDir = path.resolve(trainDir, modelName);
  const modelModelDir = path.resolve(modelDir, 'model');
  const modelPath = path.resolve(modelModelDir, 'model.json');
  const requestUrl = url.pathToFileURL(modelPath).href.replace(
    '///', 
    process.platform === 'win32' ? '//' : '///'
  ); // https://stackoverflow.com/questions/57859770

  const model = await tf.loadLayersModel(requestUrl);
  return model;
};

const saveTrainModel = async(
  model: tf.LayersModel,
  gameName: string,
  modelName: string
) => {
  const gameDir = path.resolve(dataDir, gameName);
  if (!fs.existsSync(gameDir)) {
    fs.mkdirSync(gameDir);
  }
  const trainDir = path.resolve(gameDir, 'train');
  if (!fs.existsSync(trainDir)) {
    fs.mkdirSync(trainDir);
  }
  const modelDir = path.resolve(trainDir, modelName);
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir);
  }
  const modelModelDir = path.resolve(modelDir, 'model');
  if (!fs.existsSync(modelModelDir)) {
    fs.mkdirSync(modelModelDir);
  }

  const requestUrl = url.pathToFileURL(modelModelDir).href.replace(
    '///', 
    process.platform === 'win32' ? '//' : '///'
  ); // https://stackoverflow.com/questions/57859770
  await model.save(requestUrl);
};

const deleteTrain = async(
  gameName: string
) => {
  const gameDir = path.resolve(dataDir, gameName);
  const trainDir = path.resolve(gameDir, 'train');
  if (fs.existsSync(trainDir)) {
    fs.rmdirSync(trainDir, {
      recursive: true
    });
  }
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