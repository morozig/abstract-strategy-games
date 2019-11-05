export default interface Agent {
    act(): Promise<number>;
    step(enemyAction: number): void;
    init(): void;
}