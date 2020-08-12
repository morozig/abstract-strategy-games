import GameHistory from '../interfaces/game-history';
import path from 'path';
import fs from 'fs';
import * as tf from '@tensorflow/tfjs';
import url from 'url';

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

export {
    getHistories,
    getModels,
    saveHistory,
    loadHistory,
    saveModel,
    loadModel
}