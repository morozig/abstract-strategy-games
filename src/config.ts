const prod = process.env.NODE_ENV === 'production';

const config = {
    dataUrl: prod ?
        'https://storage.googleapis.com/abstract-strategy-games-data' :
        `${window.location}data`
};

export default config;
