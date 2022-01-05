'''
<Reference>
'log_Barrier_OMB' function is from:
[1] Lilian Besson, 2018, SMPyBandits, https://github.com/SMPyBandits/SMPyBandits/
'''
import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Environments import *
import queue
import decimal
from scipy.optimize import minimize_scalar



class SSUCB:
    def __init__(self,K,T,seed,Environment):
        print('SSUCB')
                    
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        n=np.zeros(K)
        mu=np.zeros(K)
        ucb=np.zeros(K)
        for t in range(T):
            if t%1000==0:
                print('Time: ',t+1)
            if t<K:
                k=t
            else:         
                k=np.argmax(ucb)    
                
            self.r_Exp[t],self.r[t]=self.Env.observe(k)
            mu[k]=(mu[k]*(n[k])+self.r[t])/(n[k]+1)                
            n[k]=n[k]+1
            ucb[k]=mu[k]+math.sqrt(2*math.log(1+(t+1)*(math.log(t+1))**2)/n[k])
            
    def rewards(self):
        return self.r_Exp  


    
class bob_rotting:
    def __init__(self,T,seed,Environment):
        print('BOB-UCB-T')
        
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        k=0
        r_his=[]
        r_gap=0
        r_gap_sum=0
        H=math.ceil(math.sqrt(T))
        J=[1/2**i for i in (np.array(range(math.ceil(math.log(T,2))))+1)]
        if T==1:
            J=[1/2]
        alpha=min(1,math.sqrt(len(J)*math.log(len(J))/((math.e-1)*math.ceil(T/H))))
        w=np.ones(len(J))
        p=np.zeros(len(J))
        for i in range(math.ceil(T/H)):
            p=(1-alpha)*w/w.sum()+alpha/len(J)
            j=np.random.choice(len(J),1,p=p)
            beta=1/2**(j+1)
            delta=beta**(1/3)
            n=0
            r_his=[]
            for t in range(i*H,min(H*(i+1),T)):
                if t%1000==0:
                    print('Time: ',t+1)
                self.r_Exp[t],self.r[t]=self.Env.observe(k)
#                 print(t,self.r_Exp[t])
                r_his.append(self.r[t])
                n=n+1
                for l in range(n):
                    h=l+1
                    ucb=np.sum(np.array(r_his)[t+1-h:t+1])/h+math.sqrt(10*math.log(H)/h)
                    if l==0:
                        ucb_opt=ucb
                    elif ucb<ucb_opt:
                        ucb_opt=ucb

                if ucb_opt<1-delta:
                    n=0
                    r_his=[]
                    k=k+1
            w[j]=w[j]*math.exp(alpha/(len(J)*p[j])*(1/2+self.r[i*H:H*(i+1)].sum()/(26*H*math.log(T)+4*math.sqrt(H*math.log(T)))))    
    def rewards(self):
        return self.r_Exp  

class exponential_rotting:
    def __init__(self,delta,gamma,T,seed,Environment):
        print('UCB-T-exp')
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        n=0
        k=0
        mu=0
        for t in range(T):
            if t%1000==0:
                print('Time: ',t+1)
                
            self.r_Exp[t],self.r[t]=self.Env.observe(k)
            n=n+1
            r_sum=0
            mu=(mu*(n-1)+self.r[t]*(1-gamma)**(-(n-1)))/n
            ucb=mu*((1-gamma)**n)+(1-gamma)*math.sqrt(((1-gamma)**(2*n)-1)/((1-gamma)**2-1))*math.sqrt(6*math.log(T))/n
            
            if ucb<1-delta:
                n=0
                mu=0
                k=k+1
            
    def rewards(self):
        return self.r_Exp  
    
    
class linear_rotting:
    def __init__(self,delta,epsilon,T,seed,Environment):
        print('UCB-T-linear')
        
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        n=0
        k=0
        mu=0
        for t in range(T):
            if t%1000==0:
                print('Time: ',t+1)
                
            self.r_Exp[t],self.r[t]=self.Env.observe(k)
            n=n+1
            mu=(mu*(n-1)+self.r[t]+epsilon*(n-1))/n
            ucb=mu-epsilon*n+math.sqrt(8*math.log(T)/n)
                
            if ucb<1-delta:
                n=0
                mu=0
                k=k+1
            
    def rewards(self):
        return self.r_Exp  
    
  