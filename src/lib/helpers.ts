const indexMax = (numbers: number[]) =>
    numbers.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);

const softMax = (numbers: number[], softer?: boolean) => {
    const boost = softer ? 10 : 100;
    const exps = numbers.map(value => Math.exp(value * boost));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sum);
};

const indexSoftMax = (numbers: number[], softer?: boolean) => {
    const softMaxed = softMax(numbers, softer);
    const bound = Math.random();
    let total = 0;
    let index = 0;
    for (let value of softMaxed) {
        total += value;
        if (total >= bound) {
            break;
        }
        index += 1;
    }
    return index;
};

const randomOf = <T>(items: T[]) => {
    const index = Math.floor(Math.random() * items.length);
    return items[index];
};

const sleep = (ms: number) => new Promise(
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

export {
    indexMax,
    randomOf,
    indexSoftMax,
    sleep,
    durationHR
};
