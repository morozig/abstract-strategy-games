import numpy as np
import gym
from sklearn.neural_network import MLPClassifier
from ttt import TicTacToeEnv
import pickle
from collections import deque

env = TicTacToeEnv(2)

n_actions = env.action_space.n
# env.render()

agent = MLPClassifier(hidden_layer_sizes=(300,300,300),
                      activation='tanh',
                      warm_start=True, #keep progress between .fit(...) calls
                      max_iter=1 #make only 1 iteration on each .fit(...)
                     )
#initialize agent to the dimension of state an amount of actions
agent.fit([env.reset()]*n_actions,range(n_actions))

def agent_policy(state):
    return agent.predict([state])[0]

env1 = TicTacToeEnv(1, agent_policy)
env2 = TicTacToeEnv(2, agent_policy)

n_actions = env1.action_space.n
# env.rend

def generate_session(env, t_max=100):
    
    states,actions = [],[]
    total_reward = 0
    
    s = env.reset()
    
    for t in range(t_max):
        
        #predict array of action probabilities
        probs = agent.predict_proba([s])[0] 
        
        a = np.random.choice(n_actions, p=probs)
        
        new_s,r,done,info = env.step(a)
        
        #record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward+=r
        
        s = new_s
        if done:
            break
    return states,actions,total_reward

bin_size = 10

def get_elite(env):
    actions_bin = []
    bin_sizes = []
    min_bin_size = 0
    all_count = 0
    good_count = 0
    elite_states = []
    elite_actions = []

    for i in range(n_actions):
        actions_bin.append(deque(maxlen=bin_size))
        bin_sizes.append(0)

    while min_bin_size < bin_size:
        states, actions, total_reward  = generate_session(env)
        all_count += 1
        if total_reward >= 1:
            good_count += 1
            for state, action in zip(states, actions):
                actions_bin[action].append(state)
                bin_sizes[action] = len(actions_bin[action])
                min_bin_size = min(bin_sizes)

    for action in range(n_actions):
        for state in actions_bin[action]:
            elite_states.append(state)
            elite_actions.append(action)

    elite_states = np.array(elite_states)
    elite_actions = np.array(elite_actions)

    rate = good_count / all_count
    return elite_states, elite_actions, rate

for i in range(1000):

    elite_states1, elite_actions1, rate1 = get_elite(env1)
    elite_states2, elite_actions2, rate2 = get_elite(env2)

    elite_states = np.concatenate([elite_states1, elite_states2])
    elite_actions = np.concatenate([elite_actions1, elite_actions2])

    agent.fit(elite_states, elite_actions)
    print("{0:.2f} {1:.2f}".format(rate1, rate2))
    if min(rate1, rate2) >= 0.9:
        print("You won in", i, "turns!")
        break

def save():
    with open('agent.pickle', 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)
save()





