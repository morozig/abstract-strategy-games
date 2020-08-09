import url from 'url';

const prod = process.env.NODE_ENV === 'production';
const isNode = typeof window === 'undefined';

const config = {
    dataUrl: prod ?
        'https://storage.googleapis.com/abstract-strategy-games-data' :
        isNode ?
            `${
                url.pathToFileURL(process.cwd())
                    .href
                    .replace('///', '//')
                    // https://stackoverflow.com/questions/57859770
            }/data` :
            `${window.location}data`
};

export default config;
