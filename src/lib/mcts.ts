const DEFAULT_CPUCT = 5;

class Node {
    numVisits: number;
    totalValue: number;
    meanValue: number;
    probability: number | undefined;
    parent: Node | null;
    children: Node[];
    env;
    policyValueFn;
    action: number | undefined;
    state;
    reward: number;
    done: boolean;
    constructor(
        parent: Node | null,
        action?: number,
        probability?: number,
        env?,
        policyValueFn?
    ) {
        if (parent) {
            this.env = parent.env;
            this.policyValueFn = parent.policyValueFn;
        }
        {
            if (env) this.env = env;
            if (policyValueFn) this.policyValueFn = policyValueFn;
        }
        this.parent = parent;
        this.action = action;
        this.probability = probability;
        this.numVisits = 0;
        this.totalValue = 0;
        this.meanValue = 0;
        this.children = [];
        if (!parent) {
            this.state = this.env.init();
        } else {
            const [state, reward, done] = this.env.step(parent.state, action);
            this.state = state;
            this.reward = reward;
            this.done = done;
        }
    }
    findBestLeaf(cPuct = undefined): Node {
        if (this.isLeaf()) {
            return this;
        }
        let bestChild;
        let bestValue;
        for (let node of this.children) {
            const sign = ( this.state.player === node.state.player ) ?
                1 : -1;
            const value = sign * node.meanValue + node.getBonus(cPuct);
            if (!bestValue || value > bestValue) {
                bestChild = node;
                bestValue = value;
            }
        }
        return bestChild.findBestLeaf(cPuct);
    }
    isLeaf() {
        return !this.children.length;
    }
    isRoot() {
        return !this.parent;
    }
    getBonus(cPuct?: any) {
        // usb1: bonus = scale * math.sqrt((2 * math.log(self.parent.times_visited))/self.times_visited)

        const parentVisits = this.parent ? this.parent.numVisits : this.numVisits;
        const probability = this.probability || 1;
        const bonus = cPuct * probability * Math.sqrt(
            parentVisits
        ) / ( this.numVisits + 1 );
        return bonus;
    }
    propagate(value: number) {
        this.numVisits += 1;
        this.totalValue += value;
        this.meanValue = this.totalValue / this.numVisits;
        if (this.parent) {
            const sign = ( this.parent.state.player === this.state.player ) ?
                1 : -1;
            this.parent.propagate(value * sign);
        }
    }
    async expand() {
        const [ policy, value ] = await this.policyValueFn(this.state);
        for (let [ action, prob ] of policy) {
            this.children.push(new Node(
                this,
                action,
                prob
            ));
        }
        return value;
    }
}

class MCTS {
    root: Node;
    env;
    cPuct;
    // policyValueFn;
    constructor(env, policyValueFn, options?) {
        this.env = env;
        this.root = new Node(null, undefined, undefined, env, policyValueFn);
        if ( options && options.cPuct ) {
            this.cPuct = options.cPuct;
        } else {
            this.cPuct = DEFAULT_CPUCT;
        }
    }
    getAction() {
        let actionMax;
        let numVisitsMax = 0;
        for (let node of this.root.children) {
            const numVisits = node.numVisits;
            if (numVisits > numVisitsMax) {
                actionMax = node.action;
                numVisitsMax = numVisits;
            }
        }
        return actionMax;
    }
    getProbs() {
        const probs = this.root.children.map(child => [
            child.action,
            child.numVisits / this.root.numVisits
        ]);
        return probs;
    }
    step(action: number) {
        let child;
        if (!this.root.children.length) {
            if (!this.env.availables(this.root.state).includes(action)) {
                throw 'Not valid action!!';
            }
            this.root.children.push(new Node(
                this.root,
                action,
                undefined
            ));
        }
        child = this.root.children.find(
            node => node.action === action
        );
        if (!child) {
            throw 'Not valid action!!';
        }
        this.root = child;
        this.root.parent = null;
    }
    async plan(numIterations: number, cPuct = undefined) {
        if (!cPuct) cPuct = this.cPuct;
        for (let i = 0; i < numIterations; i++) {
            const node = this.root.findBestLeaf(cPuct);
            let value = 0;
            if (node.done) {
                value = node.reward;
            } else {
                value = await node.expand();
            }
            node.propagate(value);
        }
    }
}

export default MCTS;