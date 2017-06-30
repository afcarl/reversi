import numpy as np
from numpy.random import *
import chainer
from chainer import cuda,Function,gradient_check,Variable
from chainer import optimizers,serializers,utils
from chainer import Link,Chain,ChainList
import chainer.functions as F
import chainer.links as L
import random
import copy

gamma = 0.95

class Agent(chainer.Chain):
    def __init__(self):
        super(Agent, self).__init__(
        l1=L.Linear(64, 2000),
        l2=L.Linear(2000, 2000),
        l3=L.Linear(2000, 64),
        )

    def forward(self,x):
        x = x.reshape((1,64))
        x = Variable(x)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = self.l3(y)
        return y

    def backward(self, ob ,last_ob,action,r):
        q_dash = self.forward(last_ob)
        q = self.forward(ob)
        target = np.array(copy.deepcopy(q.data),dtype=np.float32)
        #print(r,gamma,np.amax(q_dash.data))
        target[0][action] = r + gamma * np.amax(q_dash.data)
        td = Variable(target) - q
        td_tmp =  td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)
        zero_val = Variable(np.zeros((1, 64), dtype=np.float32))
        model.cleargrads()
        loss = chainer.functions.mean_squared_error(td_clip, zero_val)
        loss.backward()
        optimizer.update()

model = Agent()
optimizer = optimizers.Adam()
optimizer.setup(model)
