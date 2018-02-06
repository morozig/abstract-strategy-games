import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
import copy

class TicTacToeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'ansi']
    }

    def __init__(self, player = 2, enemy_policy = None):
        self.player = player
        self.opponent = 3 - player
        if not enemy_policy:
            self.enemy_policy = TicTacToeEnv.default_policy
        else:
            self.enemy_policy = enemy_policy
        self.board_size = 3
        self.tiles = self.board_size ** 2
        self.action_space = spaces.Discrete(self.tiles)
        self.observation_space = spaces.Box(
            np.zeros(self.tiles * 3 + 2),
            np.ones(self.tiles * 3 + 2)
        )
        self.state = None
        self.done = False
        self.reward = 0

    def _reset(self):
        board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        state = TicTacToeEnv.board_to_state(board)
        state = TicTacToeEnv.set_player(state, self.player)
        if self.player == 2:
            action = self.enemy_policy(state)
            if not TicTacToeEnv.valid_move(board, action):
                self.enemy_policy = TicTacToeEnv.default_policy
                action = self.enemy_policy(state)
            board = TicTacToeEnv.make_move(board, 1, action)
            state = TicTacToeEnv.board_to_state(board)
        self.state = state
        self.done = False
        self.reward = 0
        return np.copy(state)

    def _step(self, action):
        state = np.copy(self.state)
        if self.done:
            return state, self.reward, True, {'state': self.state}
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        board = TicTacToeEnv.state_to_board(state)
        if not TicTacToeEnv.valid_move(board, action):
            return self.state, -1.0, True, {'state': self.state}
        board = TicTacToeEnv.make_move(board, self.player, action)
        done = TicTacToeEnv.is_game_finished(board)
        if done:
            self.done = True
            if done == self.opponent:
                self.reward = 0
            elif done == self.player:
                self.reward = 2
            else:
                self.reward = 1
        else:
            action = self.enemy_policy(state)
            if not TicTacToeEnv.valid_move(board, action):
                self.enemy_policy = TicTacToeEnv.default_policy
                action = self.enemy_policy(state)
            board = TicTacToeEnv.make_move(board, self.opponent, action)
            done = TicTacToeEnv.is_game_finished(board)
            if done:
                self.done = True
                if done == self.opponent:
                    self.reward = 0
                elif done == self.player:
                    self.reward = 2
                else:
                    self.reward = 1
        self.state = TicTacToeEnv.board_to_state(board)
        state = np.copy(self.state)
        return state, self.reward, self.done, {}

    def _render(self, mode='ansi', close=False):
        if close:
            return
        board = TicTacToeEnv.state_to_board(self.state)
        board_size = len(board)
        for i in range(board_size):
            print('|', end='')
            for j in range(board_size):
                tile = board[i][j]
                char = ' '
                if tile == 1:
                    char = 'X'
                elif tile == 2:
                    char = 'O'
                print(char, end='')
            print('|')

    @staticmethod
    def board_to_state(board):
        board_size = len(board)
        tiles = board_size ** 2
        state = np.zeros(tiles * 3 + 2)
        for i in range(board_size):
            for j in range(board_size):
                tile = board[i][j]
                position = tile * tiles + i * board_size + j
                state[position] = 1
        return state
    
    @staticmethod
    def state_to_board(state):
        board_size = math.floor(math.sqrt((state.size - 2) // 3))
        tiles = board_size ** 2
        board = []
        for i in range(board_size):
            board.append([])
            for j in range(board_size):
                board[i].append(0)
        for position in range(state.size - 2):
            if not state[position]:
                continue
            tile = position // tiles
            i = (position % tiles) // board_size
            j = (position % tiles) % board_size
            board[i][j] = tile
        return board
    
    @staticmethod
    def set_player(state, player):
        new_state = np.copy(state)
        if player == 1:
            new_state[new_state.size - 2] = 1
            new_state[new_state.size - 1] = 0
        else:
            new_state[new_state.size - 2] = 0
            new_state[new_state.size - 1] = 1
        return new_state

    @staticmethod
    def valid_move(board, action):
        board_size = len(board)
        i = action // board_size
        j = action % board_size
        return board[i][j] == 0
    
    @staticmethod
    def make_move(board, player, action):
        board_size = len(board)
        i = action // board_size
        j = action % board_size
        new_board = copy.deepcopy(board)
        new_board[i][j] = player
        return new_board
    
    @staticmethod
    def all_same(l):
        tile = l[0]
        for element in l:
            if element != tile:
                return None
        return tile

    @staticmethod
    def is_game_finished(board):
        board_size = len(board)
        for i in range(board_size):
            if TicTacToeEnv.all_same(board[i]):
                return board[i][0]
        for j in range(board_size):
            column = [board[i][j] for i in range(board_size)]
            if TicTacToeEnv.all_same(column):
                return column[0]
        diagonal1 = [board[i][i] for i in range(board_size)]
        if TicTacToeEnv.all_same(diagonal1):
                return diagonal1[0]
        diagonal2 = [board[board_size - 1 - i][i] for i in range(board_size)]
        if TicTacToeEnv.all_same(diagonal2):
                return diagonal2[0]
        done = all([all(board[i]) for i in range(board_size)])
        return 3 if done else 0

    @staticmethod
    def default_policy(state):
        board = TicTacToeEnv.state_to_board(state)
        board_size = len(board)
        empty = []
        action = 0
        for i in range(board_size):
            for j in range(board_size):
                tile = board[i][j]
                if tile == 0:
                    empty.append((i, j))
        if len(empty) <= 0:
            return None
        else:
            i, j = empty[np.random.choice(len(empty))]
            action = i * board_size + j
        return action

if __name__ == '__main__':
    board = [
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 1]
    ]
    state = TicTacToeEnv.board_to_state(board)
    print(state)
    print(TicTacToeEnv.state_to_board(state))
    # test_env = TicTacToeEnv()
    # test_env.reset()
    # test_env.render()

    play = True
    if play:
        play_env = TicTacToeEnv(2)
        s = play_env.reset()
        play_env.render()
        r = -1
        done = False
        info = {}
        while not done:
            i = int(input('i (1 - 3): ')) - 1
            j = int(input('j (1 - 3): ')) - 1
            action = i * 3 + j
            s, r, done, info = play_env.step(action)
            print(r)
            play_env.render()
        print(r)