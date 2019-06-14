class Channel {
    private listeners;
    constructor(){
        this.listeners = [];
    }
    set(value) {
        for (let resolve of this.listeners) {
            setTimeout(() => {
                resolve(value);
            }, 0);
        }
        this.listeners = [];
    }
    get() {
        return new Promise(resolve => {
            this.listeners.push(resolve);
        });
    }
}

export default Channel;