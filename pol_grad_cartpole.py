import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
from policy_gradient_agent import Agent

env = gym.make('CartPole-v0')

myAgent = Agent(lr=1e-2, s_size=4, a_size=2, h_size=8, b_size=10, gamma=.99) #Load the agent.

total_episodes = 8000 #Set total number of episodes to train agent on.
max_ep_length = 201

i = 0
total_reward = []
avg_rewards = []

for i in range(total_episodes):
	s = env.reset()
	running_reward = 0
	for j in range(max_ep_length):
		#Probabilistically pick an action given our network outputs.
		a = myAgent.chooseAction(s)
		#if i%1000 == 0: env.render()
		s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
		myAgent.observe(s, a, r, d)
		s = s1
		running_reward += r
		if d == True:
			total_reward.append(running_reward)
			break

	#Update our running tally of scores.
	if i % 100 == 0:
		avg_rewards.append(np.mean(total_reward[-100:]))
		print(str((i/total_episodes)*100) + "%")
		
avgX = np.linspace(0, len(total_reward), len(avg_rewards))
plt.plot(total_reward)
plt.plot(avgX, avg_rewards)
plt.show()