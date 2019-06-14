class Agent {
    channel;
    constructor(player: number, channel) {
        this.channel = channel;
    }
    act() {
        return this.channel.get();
    }
    step(action: number) {
    }
}

export default Agent;