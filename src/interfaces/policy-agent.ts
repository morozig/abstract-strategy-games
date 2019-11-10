import Agent from './agent';
import PolicyAction from './policy-action';

export default interface PolicyAgent extends Agent {
    policyAct(): Promise<PolicyAction>;
}