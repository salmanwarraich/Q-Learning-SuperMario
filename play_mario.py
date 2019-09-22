# -*- coding: utf-8 -*-
import pdb
import numpy as np
import random
import math
import cv2
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from net_pytorch import dqn_net
from replay_memory import replay_memory
from data import env
import matplotlib
import time

# if gpu is to be used
'''
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

Tensor = FloatTensor
'''
def ob_process(frame):
    '''
    Parameters
    ----------
    frame: {ndarray} of shape (90,90)

    Returns
    -------
    frame: {Tensor} of shape torch.Size([1,84,84])
    '''
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame=frame.astype('float64')
    frame=torch.from_numpy(frame)
    frame=frame.unsqueeze(0).type(Tensor)
    return frame

def plot_graph(mean_reward_list):

    plt.figure(1)
    plt.clf()
    plt.title('Episode Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(mean_reward_list)
    plt.pause(0.001)  # pause a bit so that plots can be updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def learn(env,
        MAX_EPISODE,
        EPS_START,
        EPS_END,
        EPS_DECAY,
        LEARNING_RATE,
        GAMMA,
        ):
    
    mapping_reduced_action = [3,7,11,4,10]
    Un_mapping_reduced_action = [100,100,100,0,3,100,100,1,100,100,4,2]
    ### initialization ###
    env.reset()
    obs,_,_,_,_,_,_,_,_,_=env.step(0)
    judge_distance=0
    episode_total_reward = 0
    no_states_observed = 1
    epi_total_reward_list=[]
    mean_reward_list=[]
    filename = 'State_Q_Table.csv'
    LEARNING_RATE_CTR = [np.zeros(6), np.zeros(6)]
    try:
        state_table = np.loadtxt(filename, delimiter=",", usecols=[0])
        state_table = state_table.astype(int)
        state_table = state_table.tolist()
        q_table = np.loadtxt(filename, delimiter=",", usecols=[1,2,3,4,5,6])
        LEARNING_RATE_CTR = q_table * 0
        q_table = q_table.tolist()
        LEARNING_RATE_CTR = LEARNING_RATE_CTR.tolist()
        no_states_observed = len(state_table) - 1
    except:
        print ('warning: Error %s: Loading State, Action Table' % filename)
        state_table = [0, 1]
        q_table = [np.random.rand(6), np.random.rand(6)]
    if(state_table == [] or q_table == []):
        state_table = [0, 1]
        q_table = [np.random.rand(6), np.random.rand(6)]
        LEARNING_RATE_CTR = [np.zeros(6), np.zeros(6)]
    # counters #
    time_step=0
    update_times=0
    episode_num=0
    history_distance=200
    index_s = 0
    state_d_current = state_table[index_s]
    f_handle = open(filename, 'w')
    f_handle_Evo = open('State_Q_Table_ev.csv', 'a')
    controller_speed_ctr = 0
    reward_collection = 0
    print(state_table)
    print(q_table)
    while episode_num <= MAX_EPISODE:
        ### choose an action with epsilon-greedy ###
        prob = random.random()
        threshold = EPS_END # + (EPS_START - EPS_END) * math.exp(-1 * episode_num / EPS_DECAY)
        #action_onehot = action_space[0][1] # {Tensor}
        
        #if(controller_speed_ctr == 0):
        reward_collection = 0
        if prob <= threshold:
            action_button_d = np.random.randint(6)
        else:
            action_button_d = np.argmax(q_table[index_s])
        np.savetxt(f_handle_Evo, [np.concatenate([[state_d_current,action_button_d], q_table[index_s]])], fmt = '%1.6f' ,delimiter = ',')
        
        obs_next, reward, done, _, max_distance, _, now_distance, reward_d, state_d_next, keyboard_keys = env.step(action_button_d)
        reward_collection += reward_d
        obs_next, reward, done, _, max_distance, _, now_distance, reward_d, state_d_next, keyboard_keys = env.step(action_button_d)
        reward_collection += reward_d
        obs_next, reward, done, _, max_distance, _, now_distance, reward_d, state_d_next, keyboard_keys = env.step(action_button_d)
        reward_collection += reward_d
        obs_next, reward, done, _, max_distance, _, now_distance, reward_d, state_d_next, keyboard_keys = env.step(action_button_d)
        reward_collection += reward_d

        if state_d_next in state_table:
            #start = time.clock()
            next_index = state_table.index(state_d_next)
            current_index = state_table.index(state_d_current)
            current_value = q_table[current_index][action_button_d]
            LEARNING_RATE_CTR[current_index][action_button_d] += 1
            LEARNING_RATE_S_A = LEARNING_RATE / LEARNING_RATE_CTR[current_index][action_button_d]
            #print(LEARNING_RATE_S_A)
            q_table[current_index][action_button_d] = current_value + LEARNING_RATE_S_A * (reward_d + GAMMA*(max(q_table[next_index])) - current_value)
            #print(current_value + LEARNING_RATE * (reward_d + GAMMA*(max(q_table[next_index])) - current_value))
            index_s = next_index
            #print(q_table[current_index])
            #print(current_index)
            #print(time.clock() - start)
            #print(np.concatenate(([state_table[current_index]], [reward_d], q_table[current_index])))
        else:
            state_table.append(state_d_next)
            q_table.append(np.random.rand(6))
            LEARNING_RATE_CTR.append(np.zeros(6))
            no_states_observed = len(state_table) - 1 #no_states_observed + 1
            index_s = no_states_observed
            #print(no_states_observed)
            print(np.concatenate(([state_table[no_states_observed]], [reward_d], q_table[no_states_observed])))
        state_d_current = state_d_next

        episode_total_reward +=reward_d
        if now_distance <= history_distance:
            judge_distance+=1
        else:
            judge_distance=0
            history_distance=max_distance

        ### go to the next state ###
        if done == False:
            #obs4 = obs4_next
            time_step += 1
        elif done == True or judge_distance > 50:
            env.reset()
            obs, _, _, _, _, _, _, _, _, _ = env.step(0)
            episode_num += 1
            history_distance = 200
            # plot graph #
            epi_total_reward_list.append(episode_total_reward)
            mean100=np.mean(epi_total_reward_list[-101:-1])
            mean_reward_list.append(mean100)
            plot_graph(epi_total_reward_list)
            print('episode %d total reward=%.2f'%(episode_num,episode_total_reward))
            episode_total_reward = 0
    np.savetxt(f_handle, np.column_stack((state_table, q_table)), fmt=','.join(['%i'] + ['%1.6f']*6), delimiter=',')
    np.savetxt('Reward.csv',epi_total_reward_list, fmt='%1.6f')
    np.savetxt('LR_CTR.csv',LEARNING_RATE_CTR, fmt='%i', delimiter=',')
    f_handle.close()
    f_handle_Evo.close()

if __name__=='__main__':
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    learn(env=env.Env(),
        MAX_EPISODE=30,
        EPS_START=0.01, #0.3, # 0.9
        EPS_END=0.01, #0.3, # 0.05
        EPS_DECAY=200,
        LEARNING_RATE=0.0, #0.8, # 1e-3
        GAMMA=0.6,
        )
    '''
    learn(env=env.Env(),
        MAX_EPISODE=2000000,
        EPS_START=0.3, # 0.9
        EPS_END=0.1, # 0.05
        EPS_DECAY=200,
        ACTION_NUM=6,
        REPLAY_MEMORY_CAPACITY=10000,
        BATCH_SIZE=32,
        LOSS_FUNCTION=nn.SmoothL1Loss,
        OPTIM_METHOD=optim.Adam,
        LEARNING_RATE=5e-1, # 1e-3
        GAMMA=0.99,
        NET_COPY_STEP=1000, # 1000
        OBSERVE=10000, # 10000
        TRAIN_FREQ=4,
        PATH='net_param.pt'
        )
    '''







