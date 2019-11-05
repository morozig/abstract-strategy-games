import Agent from '../interfaces/agent';

export default class Human implements Agent{
    private humanAction: () => Promise<number>;
    constructor(humanAction: () => Promise<number>) {
        this.humanAction = humanAction;
    }
    act() {
        return this.humanAction();
    }
    step() {
    }
    init() {
    }
}