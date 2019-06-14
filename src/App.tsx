import * as React from 'react';
import './App.css';
import Board from './Board';
import { Env } from './game/env';
import Channel from './lib/channel';
import Train from './game/train';
import Play from './game/play';


interface AppState {
    player: number;
    board: number[][];
    done: boolean;
    secondsPerTurn: string;
}

class App extends React.Component<any, AppState> {
    train;
    play;
    channel;
    constructor() {
        super(undefined);
        this.channel = new Channel();
        const { player, board } = Env.init();
        const done = false;
        let secondsPerTurn;
        this.state = { player, board, done, secondsPerTurn };
        this.train = new Train((state) => {
            this.setState(state);
        });
        this.play = new Play((state) => {
            this.setState(state);
        }, this.channel);
        this.train.run();
        // this.play.run();
    }
    onAction = (action: number) => {
        this.channel.set(action);
    }
    render() {
        return (
            <div className="App">
                <Board board = {this.state.board}
                    onAction = {this.onAction}
                    />
                <div>
                    <p>
                        {this.state.secondsPerTurn}
                    </p>
                </div>
            </div>
        );
    }
}

export default App;
