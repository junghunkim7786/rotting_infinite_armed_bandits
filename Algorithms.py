import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Environment import *




    
class UCB_TP:
    def __init__(self,delta,epsilon,T,seed,Environment):
        print('UCB-TP')
        
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
    

class AUCB_TP:
    def __init__(self,T,seed,Environment):
        print('AUCB_TP')
        
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        if T==1:
            k=0
            self.r_Exp[0],self.r[0]=self.Env.observe(k)
        else:
            k=0
            H=math.ceil(math.sqrt(T))
            
            B=math.ceil((3/2)*math.log(H,2))-2 
            alpha=min(1,math.sqrt(B*math.log(B)/((math.e-1)*math.ceil(T/H))))
            w=np.ones(B)
            p=np.zeros(B)
            for i in range(math.ceil(T/H)):
                p=(1-alpha)*w/w.sum()+alpha/B
                j=np.random.choice(B,1,p=p)
                beta=(1/2)**(j+3)
                delta=beta**(1/3)
                n=0
                mu=0
                for t in range(i*H,min(H*(i+1),T)):
                    if t%1000==0:
                        print('Time: ',t+1)
                    self.r_Exp[t],self.r[t]=self.Env.observe(k)
                    n=n+1
                    mu=(mu*(n-1)+self.r[t]+beta*(n-1))/n
                    ucb=mu-beta*n+math.sqrt(8*math.log(T)/n)

                    if ucb<1-delta:
                        n=0
                        mu=0
                        k=k+1

                w[j]=w[j]*math.exp(alpha/(B*p[j])*(1/2+self.r[i*H:H*(i+1)].sum()/(186*H*math.log(T)+4*math.sqrt(H*math.log(T)))))    
                
    def rewards(self):
        return self.r_Exp 
    
    
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
