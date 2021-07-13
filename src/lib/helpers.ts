const indexMax = (numbers: number[]) =>
  numbers.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);

const softMax = (numbers: number[], temp = 1) => {
  if (temp === 0 ){
    const maxI = indexMax(numbers);
    return numbers.map((_, i) => i === maxI ? 1 : 0);
  } else {
    const boost = 1 / temp;
    const exps = numbers.map(value => Math.exp(value * boost));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sum);
  }
};

const chooseIndex = (probs: number[]) => {
  const bound = Math.random();
  let total = 0;
  let index = 0;
  for (let value of probs) {
    total += value;
    if (total >= bound) {
      break;
    }
    index += 1;
  }
  return index < probs.length ?
    index :
    probs.length - 1;
}

const indexSoftMax = (numbers: number[], temp?: number) => {
  return temp === undefined ?
    chooseIndex(numbers) :
    chooseIndex(softMax(numbers, temp));
};

const randomOf = <T>(items: T[]) => {
  const index = Math.floor(Math.random() * items.length);
  return items[index];
};

const sleep = (ms: number) => new Promise<void>(
  (resolve: () => void) => setTimeout(resolve, ms)
);

const durationHR = (ms: number) => {
  const numberEnding = (num: number) => (num > 1) ? 's' : '';

  let temp = Math.floor(ms / 1000);
  const years = Math.floor(temp / 31536000);
  if (years) {
    return years + ' year' + numberEnding(years);
  }

  const days = Math.floor((temp %= 31536000) / 86400);
  if (days) {
    return days + ' day' + numberEnding(days);
  }
  const hours = Math.floor((temp %= 86400) / 3600);
  if (hours) {
    return hours + ' hour' + numberEnding(hours);
  }
  const minutes = Math.floor((temp %= 3600) / 60);
  if (minutes) {
    return minutes + ' minute' + numberEnding(minutes);
  }
  const seconds = temp % 60;
  if (seconds) {
    return seconds + ' second' + numberEnding(seconds);
  }
  return 'less than a second';
};

const cpusCount = async () => {
  if (typeof window === 'undefined') {
    const os = await import('os');
    return os.cpus().length;
  } else {
    return window.navigator.hardwareConcurrency;
  }
};

const oneHot = (action: number, actionsCount: number) => {
  const policy = new Array(actionsCount).fill(0);
  policy[action - 1] = 1;
  return policy;
};
interface RetryOptions<T> {
  
};

const retry = async <T>(
  fn: () => Promise<T>,
  retries = 3,
  sleepMs = 30 * 1000,
) => {
  let lastErr: any;

  for (let i = 0; i < retries; i++) {
    if (lastErr) {
      await sleep(sleepMs);
    }
    try {
      const result = await fn();
      return result;
    }
    catch (err) {
      console.log(`Error:, ${err}`, 'going to retry...');
      lastErr = err;
    }
  }
  throw(lastErr);
};

export {
  indexMax,
  softMax,
  randomOf,
  indexSoftMax,
  sleep,
  durationHR,
  cpusCount,
  oneHot,
  chooseIndex,
  retry
};
