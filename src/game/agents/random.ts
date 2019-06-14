import { Env } from '../env';

class Agent {
    state;
    constructor() {
        this.init();
    }
    act() {
        const availables = Env.availables(this.state);
        const index = Math.floor(Math.random() * availables.length);
        const action = availables[index];
        this.state = Env.step(this.state, action)[0];
        return action;
    }
    step(action: number) {
        this.state = Env.step(this.state, action)[0];
    }
    init() {
        this.state = Env.init();
    }
}

export default Agent;