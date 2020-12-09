const ThreadsPlugin = require('threads-plugin');

module.exports = function override(config, env) {
  config.output.globalObject = 'this'
  config.module.rules[0].parser.requireEnsure = true

  if (!config.plugins) {
    config.plugins = [];
  }

  config.plugins.push(
    new ThreadsPlugin()
  );

  return config;
};