const { execSync: cmd } = require('child_process');
const path = require('path');

const rootDir = process.cwd();
const cudaDir = path.resolve(rootDir, 'cuda');

cmd(`npm i`, {
    cwd: cudaDir
});
cmd(`npx tsc -p ${
    path.resolve(cudaDir, 'tsconfig.json')
}`);

