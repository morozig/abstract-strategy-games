const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

process.on('unhandledRejection', err => {
  throw err;
});

const rootDir = process.cwd();
const cudaDir = path.resolve(rootDir, 'cuda');

if (!fs.existsSync(path.join(rootDir, 'node_modules'))) {
  spawnSync('npm', ['i'], {
    cwd: rootDir,
    shell: process.platform === 'win32',
    stdio: 'inherit'
  });
}
if (!fs.existsSync(path.join(cudaDir, 'node_modules'))) {
  spawnSync('npm', ['i'], {
    cwd: cudaDir,
    shell: process.platform === 'win32',
    stdio: 'inherit'
  });
}


console.log('compiling typescript files');
spawnSync('npx', ['tsc', '-p', 'cuda/tsconfig.json'], {
  cwd: rootDir,
  shell: process.platform === 'win32',
  stdio: 'inherit'
});

console.log('substituting imports for node');
try {
  const replace = require('replace-in-file');
  const tfjs = replace.sync({
    files: `${cudaDir}/build/**/*.js`,
    from: '@tensorflow/tfjs',
    to: '@tensorflow/tfjs-node-gpu'
  });
  // console.log('@tensorflow/tfjs -> @tensorflow/tfjs-node-gpu', tfjs
  //   .filter(result => result.hasChanged)
  //   .map(result => result.file)
  // );
  const api = replace.sync({
    files: `${cudaDir}/build/**/*.js`,
    from: '/lib/api',
    to: '/lib/api-node'
  });
  // console.log('/lib/api -> /lib/api-node', api
  //   .filter(result => result.hasChanged)
  //   .map(result => result.file)
  // );
  const css = replace.sync({
    files: `${cudaDir}/build/**/*.js`,
    from: /require.+css.*/,
    to: '//$&'
  });
  // console.log('require(*.css); -> //require(*.css);', css
  //   .filter(result => result.hasChanged)
  //   .map(result => result.file)
  // );
}
catch (error) {
  console.error('Error occurred:', error);
}

console.log('starting training');
spawnSync('node', ['cuda/build/cuda/index.js'], {
  cwd: rootDir,
  shell: process.platform === 'win32',
  stdio: 'inherit'
});
