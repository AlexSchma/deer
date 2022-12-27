import numpy as np 
import copy
import random
import matplotlib.pyplot as plt
    episodes = 100
    length   = 5
    alpha = 0.9
    beta = 0.9
    epsilon = 0.1
    
length_list = [5,10]
for length in length_list:
    for alpha in range(0.2,1.2,0.2):     
        plot_name = "a_{}_b_{}_eps_{}_l_{}_episodes_{}.png".format(alpha,beta,epsilon,length,episodes)
        
        a1_col = np.zeros(length)
        a2_col = np.zeros(length)
        
        cols=[a1_col,a2_col]
        
        Ta1 = np.diag(np.ones(length-1),k=1)
        Ta1[-1][-1]=1
        Ta2 = np.zeros((length,length-1))
        Ta2 = np.hstack((np.ones(length).reshape((length,1)),Ta2))
        
        Tas = [Ta1,Ta2]
        
        ra1 = np.zeros(length)
        ra1[-1] =1
        ra2     = np.zeros(length)
        ra2[0]  = 0.2
        
        rs = [ra1,ra2]
        
        
        plta1 = copy.deepcopy(a1_col)
        plta2 = copy.deepcopy(a2_col)
        
        def eps_greedy(cols,x,eps=0.1):
        if random.random()<eps:
            return random.randint(0,1)
        else:
            return np.argmax((cols[0][x],cols[1][x]))
        
        for i in range(episodes):
        old_cols = copy.deepcopy(cols)
        for x in range(length):
            svector = np.zeros(length)
            svector[x] = 1
            a = eps_greedy(cols,x,epsilon)
            r = np.dot(rs[a],svector)
            x_prime = np.dot(svector,Tas[a])
            max_q = max(np.dot(x_prime,old_cols[0]),np.dot(x_prime,old_cols[1]))
            cols[a][x] += alpha * (r + beta*max_q - cols[a][x])
        plta1 = np.vstack((plta1,cols[0]))
        plta2 = np.vstack((plta2,cols[1]))
                          
        #print(cols)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        for i in range(length):
        episode_array = np.arange(episodes+1)
        ax1.plot(episode_array,plta1[::,i])
        ax1.legend(["state {}".format(x) for x in range(length)])
        
        ax1.set_title("Action 1")
        
        for i in range(length):
        episode_array = np.arange(episodes+1)
        ax2.plot(episode_array,plta2[::,i])
        ax2.legend(["state {}".format(x) for x in range(length)])
        
        ax2.set_title("Action 2")
        
        f.text(0.5, 0.04, 'Episode', ha='center', va='center')
        f.text(0.06, 0.5, 'Q-Value', ha='center', va='center', rotation='vertical')
        f.savefig(plot_name,)
    
    alpha = 0.9
    for epsilon in range(0.2,1.2,0.2):     
        plot_name = "a_{}_b_{}_eps_{}_l_{}_episodes_{}.png".format(alpha,beta,epsilon,length,episodes)
        
        a1_col = np.zeros(length)
        a2_col = np.zeros(length)
        
        cols=[a1_col,a2_col]
        
        Ta1 = np.diag(np.ones(length-1),k=1)
        Ta1[-1][-1]=1
        Ta2 = np.zeros((length,length-1))
        Ta2 = np.hstack((np.ones(length).reshape((length,1)),Ta2))
        
        Tas = [Ta1,Ta2]
        
        ra1 = np.zeros(length)
        ra1[-1] =1
        ra2     = np.zeros(length)
        ra2[0]  = 0.2
        
        rs = [ra1,ra2]
        
        
        plta1 = copy.deepcopy(a1_col)
        plta2 = copy.deepcopy(a2_col)
        
        def eps_greedy(cols,x,eps=0.1):
        if random.random()<eps:
            return random.randint(0,1)
        else:
            return np.argmax((cols[0][x],cols[1][x]))
        
        for i in range(episodes):
        old_cols = copy.deepcopy(cols)
        for x in range(length):
            svector = np.zeros(length)
            svector[x] = 1
            a = eps_greedy(cols,x,epsilon)
            r = np.dot(rs[a],svector)
            x_prime = np.dot(svector,Tas[a])
            max_q = max(np.dot(x_prime,old_cols[0]),np.dot(x_prime,old_cols[1]))
            cols[a][x] += alpha * (r + beta*max_q - cols[a][x])
        plta1 = np.vstack((plta1,cols[0]))
        plta2 = np.vstack((plta2,cols[1]))
                          
        #print(cols)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        for i in range(length):
        episode_array = np.arange(episodes+1)
        ax1.plot(episode_array,plta1[::,i])
        ax1.legend(["state {}".format(x) for x in range(length)])
        
        ax1.set_title("Action 1")
        
        for i in range(length):
        episode_array = np.arange(episodes+1)
        ax2.plot(episode_array,plta2[::,i])
        ax2.legend(["state {}".format(x) for x in range(length)])
        
        ax2.set_title("Action 2")
        
        f.text(0.5, 0.04, 'Episode', ha='center', va='center')
        f.text(0.06, 0.5, 'Q-Value', ha='center', va='center', rotation='vertical')
        f.savefig(plot_name,)