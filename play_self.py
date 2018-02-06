from ttt import TicTacToeEnv
import pickle

with open('agent.pickle', 'rb') as f:
    agent = pickle.load(f)

    def agent_policy(state):
        return agent.predict([state])[0]
    
    env = TicTacToeEnv(2, agent_policy)

    s = env.reset()
    env.render()
    while True:
        i = int(input('i (1 - 3): ')) - 1
        j = int(input('j (1 - 3): ')) - 1
        action = i * 3 + j
        s, r, done, info = env.step(action)
        env.render()
        print(r)
        if done:
            break



