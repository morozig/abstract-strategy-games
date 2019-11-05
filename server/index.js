const express = require('express');
const fileUpload = require('express-fileupload');
const path = require('path');
const fs = require('fs');

/**
 * @param {fileUpload.UploadedFile} file
 * @param {string} filePath
 */
const saveFile = (file, filePath) => {
    return new Promise((resolve, reject) => {
        file.mv(filePath, (err) => {
            if (err) {
                reject(err);
                return;
            }
            resolve();
        });
    });
};

const app = express();
const dataDir = path.resolve(__dirname, '../data');
const buildDir = path.resolve(__dirname, '../build');
app.use('/api', express.static(dataDir));
app.use('/', express.static(buildDir));
app.use(fileUpload());


if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir);
}

app.get('/api', (req, res) => {
    const games = fs.readdirSync(dataDir);
    return res.json(games);
});

app.get('/api/:game/history', (req, res) => {
    const game = req.params.game;
    const gameDir = path.resolve(dataDir, game);
    if (!fs.existsSync(gameDir)) {
        return res.json([]);
    }
    const history = [];
    const historyDir = path.resolve(gameDir, 'history');
    if (!fs.existsSync(historyDir)) {
        return res.json([]);
    }
    for (let historyJson of fs.readdirSync(historyDir)) {
        history.push(path.basename(historyJson, '.json'));
    }
    return res.json(history);
});

app.get('/api/:game/model', (req, res) => {
    const game = req.params.game;
    const gameDir = path.resolve(dataDir, game);
    if (!fs.existsSync(gameDir)) {
        return res.json([]);
    }
    const models = [];
    const modelsDir = path.resolve(gameDir, 'model');
    if (!fs.existsSync(modelsDir)) {
        return res.json([]);
    }
    for (let modelDir of fs.readdirSync(modelsDir)) {
        models.push(modelDir);
    }
    return res.json(models);
});

app.post('/api/:game/history', (req, res) => {
    if (!req.files || Object.keys(req.files).length === 0) {
        return res.status(400).send('No files were uploaded.');
    }
    const game = req.params.game;
    const gameDir = path.resolve(dataDir, game);
    if (!fs.existsSync(gameDir)) {
        fs.mkdirSync(gameDir);
    }
    const historyDir = path.resolve(gameDir, 'history');
    if (!fs.existsSync(historyDir)) {
        fs.mkdirSync(historyDir);
    }

    const fileName = Object.keys(req.files)[0];
    const file = req.files[fileName];
    const filePath = path.resolve(historyDir, fileName);
    file.mv(filePath, (err) => {
        if (err) return res.status(500).send(err);
        res.send('File uploaded!');
    });
});

app.post('/api/:game/model/:model', async (req, res) => {
    if (!req.files || Object.keys(req.files).length === 0) {
        return res.status(400).send('No files were uploaded.');
    }
    const game = req.params.game;
    const gameDir = path.resolve(dataDir, game);
    if (!fs.existsSync(gameDir)) {
        fs.mkdirSync(gameDir);
    }
    const modelsDir = path.resolve(gameDir, 'model');
    if (!fs.existsSync(modelsDir)) {
        fs.mkdirSync(modelsDir);
    }
    const model = req.params.model;
    const modelDir = path.resolve(modelsDir, model);
    if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir);
    }
    try {
        for (let fileName in req.files) {
            const file = req.files[fileName];
            const filePath = path.resolve(modelDir, fileName);
            await saveFile(file, filePath);
        }
        res.send(`Uploaded ${Object.keys(req.files).length} files`);
    }
    catch (err) {
        res.status(500).send(err);
    }
});

app.listen(3001);