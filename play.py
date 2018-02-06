from ttt import TicTacToeEnv
import pickle

with open('agent.pickle', 'rb') as f:
    env = TicTacToeEnv(2)
    agent = pickle.load(f)
    total_reward = 0

    s = env.reset()

    while True:
        a = agent.predict([s])[0]
        
        new_s,r,done,info = env.step(a)
        env.render()
        s = new_s
        total_reward += r
        if done:
            print(total_reward)
            break



