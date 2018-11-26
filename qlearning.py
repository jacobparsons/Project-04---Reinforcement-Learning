#The bulk of the source code was taken from https://gist.github.com/adesgautam/19580733f9ad262530181966dc6ac4fe#file-frozen-lake_q-learning-py

import gym
import numpy as np
import time, pickle, os
import matplotlib.pyplot as plt


    
def choose_action(state,epsilon):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action,gamma,lr_rate):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)
    
    
def play_episodes(environment, n_episodes, policy):
    wins = 0
    total_reward = 0

    for episode in range(n_episodes):

        terminated = False
        state = environment.reset()

        while not terminated:

            # Select best action to perform in a current state
            action = np.argmax(policy[state])

            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = environment.step(action)

            # Summarize total reward
            total_reward += reward

            # Update current state
            state = next_state

            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                wins += 1

    average_reward = total_reward / n_episodes

    return wins, total_reward, average_reward

# Start
def estimate_Q(environment,total_episodes,max_steps,epsilon,lr_rate,gamma):
    for episode in range(total_episodes):
        state = environment.reset()
        t = 0
        
        while t < max_steps:
    #        env.render()
    
            action = choose_action(state,epsilon)  
    
            state2, reward, done, info = env.step(action)  
    
            learn(state, state2, reward, action, gamma, lr_rate)
    
            state = state2
    
            t += 1
           
            if done:
                break
#    print Q
                

######################################################
#Learning graphs for 4x4 Grid
env = gym.make('FrozenLake-v0')
total_episodes = 100000
max_steps = 100

#Epsilon learning curve
epsilon_list = []
average_reward_list = []
#default epsilon 0.9
for epsilon in range(1,11):
    epsilon = epsilon/10.0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    estimate_Q(env,total_episodes,max_steps,epsilon,lr_rate=0.01,gamma=0.99)
    wins, total_reward, average_reward = play_episodes(env,5000,Q)
    epsilon_list.append(epsilon)
    average_reward_list.append(average_reward)
    print epsilon, average_reward

#Create epsilon graph
fig, ax = plt.subplots()
ax.plot(epsilon_list, average_reward_list)
ax.set(xlabel=r'$\epsilon$', ylabel='Average Reward',title='4x4 Grid: Average Reward vs. Prob. of Random Action')
ax.grid()
plt.xlim(0.1,1.0)
plt.ylim(0,1)
fig.savefig("qlearning_avgreward_epsilon_44.png")
#plt.show()
plt.clf()

#Learning rate learning curve
lr_list = []
average_reward_list = []
#default learning rate 0.01
for lr in range(1,20):
    lr = lr/20.0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    estimate_Q(env,total_episodes,max_steps,epsilon=0.9,lr_rate=lr,gamma=0.99)
    wins, total_reward, average_reward = play_episodes(env,5000,Q)
    lr_list.append(lr)
    average_reward_list.append(average_reward)
    print lr, average_reward

#Create learning rate graph
fig, ax = plt.subplots()
ax.plot(lr_list, average_reward_list)
ax.set(xlabel=r'$\alpha$', ylabel='Average Reward',title='4x4 Grid: Average Reward vs. Learning Rate')
ax.grid()
plt.xlim(0.1,1.0)
plt.ylim(0,1)
fig.savefig("qlearning_avgreward_learningrate_44.png")
#plt.show()
plt.clf()

#discount factor rate learning curve
gamma_list = []
average_reward_list = []
#default gamma 0.99
for gamma in range(1,20):
    gamma = gamma/20.0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    estimate_Q(env,total_episodes,max_steps,epsilon=0.9,lr_rate=0.01,gamma=gamma)
    wins, total_reward, average_reward = play_episodes(env,5000,Q)
    gamma_list.append(gamma)
    average_reward_list.append(average_reward)
    print gamma, average_reward

#Create learning rate graph
fig, ax = plt.subplots()
ax.plot(gamma_list, average_reward_list)
ax.set(xlabel=r'$\gamma$', ylabel='Average Reward',title='4x4 Grid: Average Reward vs. Discount Factor')
ax.grid()
plt.xlim(0.05,0.95)
plt.ylim(0,1)
fig.savefig("qlearning_avgreward_discountrate_44.png")
#plt.show()
plt.clf()


###########################################################
#Learning graphs for 8x8 Grid

env = gym.make('FrozenLake8x8-v0')
total_episodes = 100000
max_steps = 100

#Epsilon learning curve
epsilon_list_88 = []
average_reward_list_epsilon_88 = []
#default epsilon 0.9
for epsilon in range(1,11):
    epsilon = epsilon/10.0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    estimate_Q(env,total_episodes,max_steps,epsilon,lr_rate=0.01,gamma=0.99)
    wins, total_reward, average_reward = play_episodes(env,5000,Q)
    epsilon_list_88.append(epsilon)
    average_reward_list_epsilon_88.append(average_reward)
    print epsilon, average_reward

#Create epsilon graph
fig, ax = plt.subplots()
ax.plot(epsilon_list_88, average_reward_list_epsilon_88)
ax.set(xlabel=r'$\epsilon$', ylabel='Average Reward',title='8x8 Grid: Average Reward vs. Prob. of Random Action')
ax.grid()
plt.xlim(0.1,1.0)
plt.ylim(0,1)
fig.savefig("qlearning_avgreward_epsilon_88.png")
#plt.show()
plt.clf()

#Learning rate learning curve
lr_list_88 = []
average_reward_list_lr_88 = []
#default learning rate 0.01
for lr in range(1,20):
    lr = lr/20.0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    estimate_Q(env,total_episodes,max_steps,epsilon=0.9,lr_rate=lr,gamma=0.99)
    wins, total_reward, average_reward = play_episodes(env,5000,Q)
    lr_list_88.append(lr)
    average_reward_list_lr_88.append(average_reward)
    print lr, average_reward

#Create learning rate graph
fig, ax = plt.subplots()
ax.plot(lr_list_88, average_reward_list_lr_88)
ax.set(xlabel=r'$\alpha$', ylabel='Average Reward',title='8x8 Grid: Average Reward vs. Learning Rate')
ax.grid()
plt.xlim(0.1,1.0)
plt.ylim(0,1)
fig.savefig("qlearning_avgreward_learningrate_88.png")
#plt.show()
plt.clf()

#discount factor rate learning curve
gamma_list_88 = []
average_reward_list_gamma_88 = []
#default gamma 0.99
for gamma in range(1,20):
    gamma = gamma/20.0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    estimate_Q(env,total_episodes,max_steps,epsilon=0.9,lr_rate=0.01,gamma=gamma)
    wins, total_reward, average_reward = play_episodes(env,5000,Q)
    gamma_list_88.append(gamma)
    average_reward_list_gamma_88.append(average_reward)
    print gamma, average_reward

#Create learning rate graph
fig, ax = plt.subplots()
ax.plot(gamma_list_88, average_reward_list_gamma_88)
ax.set(xlabel=r'$\gamma$', ylabel='Average Reward',title='8x8 Grid: Average Reward vs. Discount Factor')
ax.grid()
plt.xlim(0.05,0.95)
plt.ylim(0,1)
fig.savefig("qlearning_avgreward_discountrate_88.png")
#plt.show()
plt.clf()

