import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from scipy.stats import bernoulli


class linear_rotting_many_Env:
    def __init__(self,epsilon,seed,T):
        np.random.seed(seed)
        self.optimal=1
        self.exp_reward=np.zeros(T)
        self.epsilon=epsilon
        self.T=T
        for k in range(self.T):
            self.exp_reward[k]=np.random.uniform(0,1)
            
#         print(self.exp_reward)
    def observe(self,k):
        reward=self.exp_reward[k]+np.random.normal(0,1)
        exp_reward=self.exp_reward[k]
        self.exp_reward[k]=self.exp_reward[k]-self.epsilon
#         print(exp_reward)
        return exp_reward,reward

#     def optimal_rewards(self):
        
#         return self.optimal

class exponential_rotting_many_Env:
    def __init__(self,gamma,seed,T):
        np.random.seed(seed)
        self.optimal=1
        self.exp_reward=np.zeros(T)
        self.gamma=gamma
        self.T=T
        for k in range(self.T):
            self.exp_reward[k]=np.random.uniform(0,1)
            
#         print(self.exp_reward)
    def observe(self,k):
        reward=self.exp_reward[k]+np.random.normal(0,1)
        exp_reward=self.exp_reward[k]
        self.exp_reward[k]=self.exp_reward[k]*(1-self.gamma)
#         print(self.exp_reward)
        return exp_reward,reward



class  Adversarial_MAB_Env:
    def __init__(self,K,S,T,seed1):
        seed(seed1)
        disturb_time=np.array([math.ceil(T*i/(S+1)) for i in range(S+1)])
        print('shift time: ',disturb_time)
        self.arms=np.zeros([T,K], float)
        self.opti_arm=np.zeros(T, int)
        for t in range(T):
            if t in disturb_time:
                if t!=0:
                    candidate=[i for i in range(K) if i != optimal_arm]
                    optimal_arm=np.random.choice(candidate) 
                else:
                    optimal_arm=np.random.choice(range(K)) 
            self.arms[t,optimal_arm]=np.random.uniform(0.6,0.95,size=1)
            for i in range(K):
                if i!=optimal_arm:
                    self.arms[t,i]=np.random.uniform(0,0.8,size=1)
        for j in range(S+1):
            a=disturb_time[j]
            if j==S:
                b=T
            else:
                b=disturb_time[j+1]
            self.opti_arm[a:b]=self.arms[a:b,:].sum(axis=0).argmax()

    
    
    def observe(self,t,a):

        reward=self.arms[t,a]
        return reward

    def optimal_rewards(self):
        T=len(self.opti_arm)
        rewards=np.zeros(T,float)
        for t in range(T):
            rewards[t]=self.arms[t,self.opti_arm[t]]
        
        return rewards
           