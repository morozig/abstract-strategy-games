import { sleep } from "./helpers";

type BatchFunc<I,O> = (inputs: I[]) => Promise<O[]>;
interface Item<I,O> {
    input: I;
    resolve: (output: O) => void;
}

export default class Batcher<I,O> {
    private func: BatchFunc<I,O>;
    private size: number;
    private wait: number;
    private onFull: (() => void) | null = null;
    private queue: Item<I,O>[] = [];
    constructor(func: BatchFunc<I,O>, size: number, wait: number) {
        this.func = func;
        this.size = size;
        this.wait = wait;
    }
    private batchFull() {
        return new Promise((resolve: () => void) => {
            this.onFull = resolve;
        });
    }
    private async callBatch() {
        await Promise.race([
            sleep(this.wait),
            this.batchFull()
        ]);
        // console.log('batch:',this.queue.length);
        const items = this.queue.slice();
        this.queue = [];
        this.onFull = null;
        const inputs = items.map(item => item.input);
        const outputs = await this.func(inputs);
        items.forEach((item, i) => {
            item.resolve(outputs[i]);
        });
    }
    private add(item: Item<I,O>) {
        this.queue.push(item);
        if (this.queue.length === 1) {
            this.callBatch();
        }
        if (this.queue.length === this.size) {
            if (this.onFull) {
                this.onFull();
            }
        }
    }
    call(inputs: I[]) {
        return Promise.all(inputs.map(
            input => new Promise((resolve: (output: O) => void) => {
                const item = {
                    input,
                    resolve
                };
                this.add(item);
            })
        ));
    }
};