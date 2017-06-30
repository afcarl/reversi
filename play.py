import numpy as np
from numpy.random import *
from reversi import Board
from agent import Agent
from tqdm import tqdm
import random
import copy
from chainer import serializers

agent = Agent()
env = Board()
env.reset()

epsilon = 0.1
gamma = 0.95
time = 0
last_Q = np.zeros((64),dtype=np.float32)
Q = np.zeros((64), dtype=np.float32)
r = np.zeros((64), dtype=np.float32)
first = True
count = 0
ave = 0.0
data = []
episode = 0
win = 0.


for e in tqdm(range(100000)):
    observation = env.reset()
    for t in range(64):
        #env.render()
        Q = agent.forward(observation)
        able = env.check_board(1)
        if(random.random()>=epsilon):
            Q_ = copy.deepcopy(Q.data[0])
            for i in range(len(Q_)):
                if i not in able:
                    Q_[i] = -100.
            action = np.argmax(Q_)
        else:
            action = random.choice(able)
        epsilon -= 5.0/10**1
        if(epsilon < 0.1):
            epsilon = 0.1

        observation, reward, done = env.step(action)
        if first:
            first=False
        else:
            agent.backward(observation, last_observation, last_action, reward)

        if done:
            episode += 1
            ave += t+1
            time += t+1
            ob = observation.reshape((1,64))
            ob = ob[0]
            if np.sum(ob) > 0:
                win += 1.
            if episode != 0 and episode % 100 == 0:
                print('episode:'+str(episode),win/100., 'epsilon:'+str(epsilon))
                serializers.save_npz("model.npz", agent)
                win = 0.
            break
        last_observation = observation
        last_action = action
