import GameHistory from '../interfaces/game-history';
import path from 'path';
import fs from 'fs/promises';

const dataDir = path.resolve(process.cwd(), '../data');

const getHistories = async (gameName: string) => {
    const histories = [] as string[];
    const gameDir = path.resolve(dataDir, gameName);
    const historyDir = path.resolve(gameDir, 'history');

    try {
        for (let historyJson of await fs.readdir(historyDir)) {
            histories.push(path.basename(historyJson, '.json'));
        }
    }
    finally {
        return histories;
    }
};

const getModels = async (gameName: string) => {
    const models = [] as string[];
    const gameDir = path.resolve(dataDir, gameName);
    const modelsDir = path.resolve(gameDir, 'model');

    try {
        for (let modelDir of await fs.readdir(modelsDir)) {
            models.push(modelDir);
        }
    }
    finally {
        return models;
    }
};

const loadHistory = async(gameName: string, historyName: string) => {
    const gameDir = path.resolve(dataDir, gameName);
    const historyDir = path.resolve(gameDir, 'history');
    const filePath = path.resolve(
        historyDir,
        `${historyName}.json`
    );

    const rawData = await fs.readFile(filePath, 'utf8');
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
    const historyDir = path.resolve(gameDir, 'history');
    const filePath = path.resolve(historyDir, historyName);
    await fs.writeFile(
        filePath,
        JSON.stringify(gameHistories),
        'utf8'
    );
};

export {
    getHistories,
    getModels,
    saveHistory,
    loadHistory
}