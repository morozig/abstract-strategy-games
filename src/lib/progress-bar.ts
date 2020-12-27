import { durationHR } from "./helpers";

const isNode = typeof window === 'undefined';

const consoleStream = {
  cursorTo() {},
  write(str: string) {
    console.log(str);
  },
  clearLine() {}
};

declare type ValueUpdater = (curr: number) => number;
declare type TokensUpdater = (tokens: any) => any;

interface ProgressBarOptions {
  curr?: number;
  total: number;
  tokens?: any;
};

class ProgressBar {
  private format: string;
  private curr: number;
  private total: number;
  private chars = {
    complete   : '=',
    incomplete : '-',
    head       : '='
  };
  private width = 40;
  private renderHandle: NodeJS.Timeout | null = null;
  private renderTimeout = isNode ? 100 : 10000;
  private startTime = 0;
  private tokens: any;
  private stream = (isNode && process.stderr.isTTY) ?
    process.stderr :
    consoleStream;
  constructor(format: string, options: ProgressBarOptions) {
    this.format = format;
    this.curr = options.curr || 0;
    this.total = options.total;
    this.tokens = options.tokens || {};
  }
  update(curr: number, tokens?: any): void;
  update(curr: ValueUpdater, tokens?: TokensUpdater): void;
  update(curr: number | ValueUpdater, tokens?: any | TokensUpdater) {
    if (typeof curr === 'number') {
      this.curr = curr;
    } else {
      this.curr = curr(this.curr);
    }
    if (tokens) {
      if (Object.keys(tokens).length) {
        this.tokens = tokens;
      } else {
        this.tokens = {
          ...this.tokens,
          ...tokens(this.tokens)
        };
      }
    }
  }
  start() {
    this.startTime = new Date().getTime();
    this.render();
    this.renderHandle = setInterval(() => this.render(), this.renderTimeout);
  }
  stop() {
    this.render();
    if (this.renderHandle) {
      clearInterval(this.renderHandle);
      this.renderHandle = null;
    }
    this.stream.write('\n');
  }
  private render() {
    let ratio = this.curr / this.total;
    ratio = Math.min(Math.max(ratio, 0), 1);

    let percent = Math.floor(ratio * 100);
    let incomplete, complete, completeLength;
    let elapsed = new Date().getTime() - this.startTime;
    let eta = (percent === 100) ? 0 : elapsed * (this.total / this.curr - 1);
    let rate = this.curr / (elapsed / 1000);

    /* populate the bar template with percentages and timestamps */
    let str = this.format
      .replace(':current', `${this.curr}`)
      .replace(':total', `${this.total}`)
      .replace(':elapsed', durationHR(elapsed))
      .replace(':eta', durationHR(eta))
      .replace(':percent', percent.toFixed(0) + '%')
      .replace(':rate', `${Math.round(rate)}`);

    let width = this.width;

    /* TODO: the following assumes the user has one ':bar' token */
    completeLength = Math.round(width * ratio);
    complete = Array(Math.max(0, completeLength + 1)).join(this.chars.complete);
    incomplete = Array(Math.max(0, width - completeLength + 1)).join(this.chars.incomplete);

    /* add head to the complete string */
    if(completeLength > 0) {
      complete = complete.slice(0, -1) + this.chars.head;
    }

    /* fill in the actual progress bar */
    str = str.replace(':bar', complete + incomplete);

    /* replace the extra tokens */
    if (this.tokens) {
      for (let key in this.tokens) {
        str = str.replace(':' + key, this.tokens[key]);
      }
    }
    this.stream.cursorTo(0);
    this.stream.write(str);
    this.stream.clearLine(1);
  }
}

export default ProgressBar;