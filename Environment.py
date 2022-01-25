import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from scipy.stats import bernoulli


class rotting_many_Env:
    def __init__(self,epsilon,seed,T):
        np.random.seed(seed)
        self.optimal=1
        self.exp_reward=np.zeros(T)
        self.epsilon=epsilon
        self.T=T
        for k in range(self.T):
            self.exp_reward[k]=np.random.uniform(0,1)
            
    def observe(self,k):
        reward=self.exp_reward[k]+np.random.normal(0,1)
        exp_reward=self.exp_reward[k]
        self.exp_reward[k]=self.exp_reward[k]-self.epsilon
        return exp_reward, reward


