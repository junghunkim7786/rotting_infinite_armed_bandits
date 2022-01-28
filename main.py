from Environment import *
from Algorithms import *
import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path



def run_T(ratio,T,num,repeat,boolean=True): #regret vs T
    T_1=int(T/num)
    num=num+1
    std_list1=np.zeros(num)
    regret_list1=np.zeros(num)
    std_list2=np.zeros(num)
    regret_list2=np.zeros(num)
    std_list3=np.zeros(num)
    regret_list3=np.zeros(num)
    T_list=np.zeros(num)
    if boolean: ##save data
        for i in range(num):
            print(i)
            if i==0:
                T=1
            else:
                T=T_1*i
            rho=1/T**ratio
            delta=rho**(1/3)
            T_list[i]=T
            K=math.ceil(math.sqrt(T)) ##number of subsampled arms in SSUCB
            regret=np.zeros(T,float)
            regret_sum=np.zeros(T,float)
            regret_sum_list1=np.zeros((repeat,T),float)
            regret_sum_list2=np.zeros((repeat,T),float)
            regret_sum_list3=np.zeros((repeat,T),float)
            std1=np.zeros(T,float)
            std2=np.zeros(T,float)
            std3=np.zeros(T,float)
            avg_regret_sum1=np.zeros(T,float)
            avg_regret_sum2=np.zeros(T,float)
            avg_regret_sum3=np.zeros(T,float)

        ###Run model
            for j in range(repeat):
                print('repeat: ',j)
                seed=j
                Env=rotting_many_Env(rho,seed,T)
                algorithm1=UCB_TP(delta,rho,T,seed,Env)
                Env=rotting_many_Env(rho,seed,T)
                algorithm2=SSUCB(K,T,seed,Env)
                Env=rotting_many_Env(rho,seed,T)
                algorithm3=AUCB_TP(T,seed,Env)
                opti_rewards=Env.optimal

                regret=opti_rewards-algorithm1.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list1[j,:]=regret_sum
                avg_regret_sum1+=regret_sum
                
                regret=opti_rewards-algorithm2.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list2[j,:]=regret_sum
                avg_regret_sum2+=regret_sum
                
                regret=opti_rewards-algorithm3.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list3[j,:]=regret_sum
                avg_regret_sum3+=regret_sum
                
            avg1=avg_regret_sum1/repeat
            sd1=np.std(regret_sum_list1,axis=0)
            avg2=avg_regret_sum2/repeat
            sd2=np.std(regret_sum_list2,axis=0)
            avg3=avg_regret_sum3/repeat
            sd3=np.std(regret_sum_list3,axis=0)


            algorithms = ['algorithm1','SSUCB','algorithm2']
            regret = dict()
            std=dict()
            regret['algorithm1']=avg1
            std['algorithm1']=sd1
            regret['SSUCB']=avg2
            std['SSUCB']=sd2
            regret['algorithm2']=avg3
            std['algorithm2']=sd3
            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]
            regret_list3[i]=avg3[T-1]
            std_list3[i]=sd3[T-1]
            
            
            ##Save data
            Path("./result").mkdir(parents=True, exist_ok=True)
            filename_1='T'+str(T)+'repeat'+str(repeat)+'ratio'+str(ratio)+'regret.txt'
            with open('./result/'+filename_1, 'wb') as f:
                pickle.dump(regret, f)
                f.close()

            filename_2='T'+str(T)+'repeat'+str(repeat)+'ratio'+str(ratio)+'std.txt'
            with open('./result/'+filename_2, 'wb') as f:
                pickle.dump(std, f)
                f.close()
    
    else: ##load data
        print('load data without running')
        for i in range(num):
            print(i)
            if i==0:
                T=1
            else:
                k=i
                T=T_1*k
            rho=1/T**ratio
            T_list[i]=T
            filename_1='T'+str(T)+'repeat'+str(repeat)+'ratio'+str(ratio)+'regret.txt'
            filename_2='T'+str(T)+'repeat'+str(repeat)+'ratio'+str(ratio)+'std.txt'
            pickle_file1 = open('./result/'+filename_1, "rb")
            pickle_file2 = open('./result/'+filename_2, "rb")
            objects = []

            while True:
                try:
                    objects.append(pickle.load(pickle_file1))
                except EOFError:
                    break
            pickle_file1.close()
            regret=objects[0]
            objects = []
            while True:
                try:
                    objects.append(pickle.load(pickle_file2))
                except EOFError:
                    break
            pickle_file2.close()
            std=objects[0]
            avg1=regret['algorithm1']
            sd1=std['algorithm1']
            avg2=regret['SSUCB']
            sd2=std['SSUCB']
            avg3=regret['algorithm2']
            sd3=std['algorithm2']
            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]
            regret_list3[i]=avg3[T-1]
            std_list3[i]=sd3[T-1]
            
    fig,(ax)=plt.subplots(1,1)

    ax.errorbar(x=T_list, y=regret_list1, yerr=1.96*std_list1/np.sqrt(repeat), color="orange", capsize=3,
                 marker="^", markersize=7,label='Algorithm 1',zorder=3) 
    ax.errorbar(x=T_list, y=regret_list3, yerr=1.96*std_list3/np.sqrt(repeat), color="b", capsize=3,
                 marker="o", markersize=7,label='Algorithm 2',zorder=2)
    ax.errorbar(x=T_list, y=regret_list2, yerr=1.96*std_list2/np.sqrt(repeat), color="g", capsize=3,
                 marker="s", markersize=7,label='SSUCB',zorder=1)
    
    #font size
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)
    # remove the errorbars in legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels,numpoints=1)
    # plot 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(r'$T$',fontsize=15)
    plt.ylabel(r'E$[R^{\pi}(T)]$',fontsize=15)
    plt.savefig('./result/T'+str(T)+'repeat'+str(repeat)+'ratio'+str(ratio)+'.png')
    plt.show()
    plt.clf()    

def run_rho(T,ratio_list,repeat,boolean=True): ## regret vs rho
    num=len(ratio_list)
    std_list1=np.zeros(num)
    regret_list1=np.zeros(num)
    std_list2=np.zeros(num)
    regret_list2=np.zeros(num)
    std_list3=np.zeros(num)
    regret_list3=np.zeros(num)
    if boolean: ##save data
        for i, ratio in enumerate(ratio_list):
            print(i)
            rho=(1/T)**ratio
            delta=rho**(1/3)
            K=math.ceil(math.sqrt(T)) ##number of subsampled arms in SSUCB
            regret=np.zeros(T,float)
            regret_sum=np.zeros(T,float)
            regret_sum_list1=np.zeros((repeat,T),float)
            regret_sum_list2=np.zeros((repeat,T),float)
            regret_sum_list3=np.zeros((repeat,T),float)
            std1=np.zeros(T,float)
            std2=np.zeros(T,float)
            std3=np.zeros(T,float)
            avg_regret_sum1=np.zeros(T,float)
            avg_regret_sum2=np.zeros(T,float)
            avg_regret_sum3=np.zeros(T,float)

        ###Run model
            for j in range(repeat):
                print('repeat: ',j)
                seed=j
                Env=rotting_many_Env(rho,seed,T)
                algorithm1=UCB_TP(delta,rho,T,seed,Env)
                Env=rotting_many_Env(rho,seed,T)
                algorithm2=SSUCB(K,T,seed,Env)
                Env=rotting_many_Env(rho,seed,T)
                algorithm3=AUCB_TP(T,seed,Env)
                opti_rewards=Env.optimal

                regret=opti_rewards-algorithm1.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list1[j,:]=regret_sum
                avg_regret_sum1+=regret_sum
                
                regret=opti_rewards-algorithm2.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list2[j,:]=regret_sum
                avg_regret_sum2+=regret_sum
                
                regret=opti_rewards-algorithm3.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list3[j,:]=regret_sum
                avg_regret_sum3+=regret_sum
                
            avg1=avg_regret_sum1/repeat
            sd1=np.std(regret_sum_list1,axis=0)
            avg2=avg_regret_sum2/repeat
            sd2=np.std(regret_sum_list2,axis=0)
            avg3=avg_regret_sum3/repeat
            sd3=np.std(regret_sum_list3,axis=0)


            algorithms = ['algorithm1','SSUCB','algorithm2']
            regret = dict()
            std=dict()
            regret['algorithm1']=avg1
            std['algorithm1']=sd1
            regret['SSUCB']=avg2
            std['SSUCB']=sd2
            regret['algorithm2']=avg3
            std['algorithm2']=sd3
            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]
            regret_list3[i]=avg3[T-1]
            std_list3[i]=sd3[T-1]
            
            ##Save data

            Path("./result").mkdir(parents=True, exist_ok=True)
            filename_1='T'+str(T)+'repeat'+str(repeat)+'ratio'+str(ratio)+'regret.txt'
            with open('./result/'+filename_1, 'wb') as f:
                pickle.dump(regret, f)
                f.close()

            filename_2='T'+str(T)+'repeat'+str(repeat)+'ratio'+str(ratio)+'std.txt'
            with open('./result/'+filename_2, 'wb') as f:
                pickle.dump(std, f)
                f.close()
    
    else: ##load data
        print('load data without running')
        for i, ratio in enumerate(ratio_list):
            print(i)
            rho=1/T**ratio
            filename_1='T'+str(T)+'repeat'+str(repeat)+'ratio'+str(ratio)+'regret.txt'
            filename_2='T'+str(T)+'repeat'+str(repeat)+'ratio'+str(ratio)+'std.txt'
            pickle_file1 = open('./result/'+filename_1, "rb")
            pickle_file2 = open('./result/'+filename_2, "rb")
            objects = []

            while True:
                try:
                    objects.append(pickle.load(pickle_file1))
                except EOFError:
                    break
            pickle_file1.close()
            regret=objects[0]
            objects = []
            while True:
                try:
                    objects.append(pickle.load(pickle_file2))
                except EOFError:
                    break
            pickle_file2.close()
            std=objects[0]
            avg1=regret['algorithm1']
            sd1=std['algorithm1']
            avg2=regret['SSUCB']
            sd2=std['SSUCB']
            avg3=regret['algorithm2']
            sd3=std['algorithm2']
            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]
            regret_list3[i]=avg3[T-1]
            std_list3[i]=sd3[T-1]
    rho_list=[(1/T)**(i/3) for i in ratio_list]
    fig,(ax)=plt.subplots(1,1)
    ax.errorbar(x=rho_list, y=regret_list1, yerr=1.96*std_list1/np.sqrt(repeat), color="orange", capsize=3,
                 marker="^", markersize=7,label='Algorithm 1',zorder=3) 
    ax.errorbar(x=rho_list, y=regret_list3, yerr=1.96*std_list3/np.sqrt(repeat), color="b", capsize=3,
                 marker="o", markersize=7,label='Algorithm 2',zorder=2)
    ax.errorbar(x=rho_list, y=regret_list2, yerr=1.96*std_list2/np.sqrt(repeat), color="g", capsize=3,
                 marker="s", markersize=7,label='SSUCB',zorder=1)
    
    #font size
    ax.tick_params(labelsize=15)
    plt.rc('legend',fontsize=15)
    ax.yaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.get_offset_text().set_fontsize(15)

    # remove the errorbars in legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels,numpoints=1)
    # plot 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(r'$\varrho^{1/3}$',fontsize=15)
    plt.ylabel(r'E$[R^{\pi}(T)]$',fontsize=15)
    plt.savefig('./result/T'+str(T)+'repeat'+str(repeat)+'ratio_list'+str(ratio_list)+'.png')
    plt.show()
    plt.clf()    

        
if __name__=='__main__':
    # Read input
    opt = int(sys.argv[1]) ##'1': a in Figure 3, '2': b,c,d in Figure 3
    
    repeat=10  # repeat number of running algorithms with different seeds.
    run_bool=True ##  True: run model and save data with plot, False: load data with plot.
    T=10**6  #Maximum Time horizon
    num=10 # number of horizon times over T
    
    if opt==1: #regret vs rho
        ratio_list=[3/10,4/10,5/10,6/10,7/10,8/10,9/10,1] #rotting rate with 1/T^ratio
        run_rho(T,ratio_list,repeat,run_bool)
        
    if opt==2: #regret vs T
        ratio=1/2  #rotting rate with 1/T^ratio
        run_T(ratio,T,num,repeat,run_bool)
        ratio=3/5
        run_T(ratio,T,num,repeat,run_bool)
        ratio=3/2
        run_T(ratio,T,num,repeat,run_bool)











