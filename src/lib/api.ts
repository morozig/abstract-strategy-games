import GameHistory from "../interfaces/game-history";

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

export {
    getHistories,
    getModels,
    saveHistory,
    loadHistory
}