import { ReactElement } from 'react';
import GameState from './game-state';

interface GameComponentProps {
    gameState: GameState;
    onAction: (action: number) => void;
}

interface GameComponent {
    (props: GameComponentProps): ReactElement
};

export default GameComponent;
