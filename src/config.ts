const prod = process.env.NODE_ENV === 'production';
const isNode = typeof window === 'undefined';
const rootLocation = isNode ?
    `file://${process.cwd()}` :
    `${window.location}`;

const config = {
    dataUrl: prod ?
        'https://storage.googleapis.com/abstract-strategy-games-data' :
        `${rootLocation}data`,
    
};

export default config;
