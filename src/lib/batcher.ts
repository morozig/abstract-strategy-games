type BatchFunc<I,O> = (inputs: I[]) => Promise<O[]>;
interface Item<I,O> {
  input: I;
  resolve: (output: O) => void;
}

export default class Batcher<I,O> {
  private func: BatchFunc<I,O>;
  private size: number;
  private wait: number;
  private timeout: NodeJS.Timeout | null = null;
  private queue: Item<I,O>[] = [];
  constructor(func: BatchFunc<I,O>, size: number, wait: number) {
    this.func = func;
    this.size = size;
    this.wait = wait;
  }
  private async callBatch() {
    const items = this.queue.slice(0, this.size);
    this.queue.splice(0, this.size);
    const inputs = items.map(item => item.input);
    const outputs = await this.func(inputs);
    items.forEach((item, i) => {
      item.resolve(outputs[i]);
    });
  }
  private add(item: Item<I,O>) {
    this.queue.push(item);
    if (this.queue.length === 1) {
      this.timeout = setTimeout(
        () => {
          this.timeout = null;
          this.callBatch();
        },
        this.wait
      );
    }
    if (this.queue.length >= this.size) {
      if (this.timeout) {
        clearTimeout(this.timeout);
        this.timeout = null;
      }
      this.callBatch();
    }
  }
  call(input: I) {
    return new Promise((resolve: (output: O) => void) => {
      const item = {
        input,
        resolve
      };
      this.add(item);
    });
  }
};