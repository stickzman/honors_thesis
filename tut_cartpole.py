import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
from tut_policy_gradient_agent import agent

try:
	xrange = xrange
except:
	xrange = range
	
env = gym.make('CartPole-v0')

gamma = 0.99

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r
	
		

tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 25

init = tf.global_variables_initializer()

avg_rewards = []

# Launch the tensorflow graph
with tf.Session() as sess:
	sess.run(init)
	i = 0
	total_reward = []
	total_length = []
		
	gradBuffer = sess.run(tf.trainable_variables())
	for ix,grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0
		
	while i < total_episodes:
		s = env.reset()
		running_reward = 0
		ep_history = []
		for j in range(max_ep):
			#Probabilistically pick an action given our network outputs.
			a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
			a = np.random.choice(a_dist[0],p=a_dist[0])
			a = np.argmax(a_dist == a)

			s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
			ep_history.append([s,a,r,s1])
			s = s1
			running_reward += r
			if d == True:
				#Update the network.
				ep_history = np.array(ep_history)
				ep_history[:,2] = discount_rewards(ep_history[:,2])
				feed_dict={myAgent.reward_holder:ep_history[:,2],
						myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
				grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
				for idx,grad in enumerate(grads):
					gradBuffer[idx] += grad
					
				if i % update_frequency == 0 and i != 0:
					feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
					sess.run(myAgent.update_batch, feed_dict=feed_dict)
					for ix,grad in enumerate(gradBuffer):
						gradBuffer[ix] = grad * 0
				
				total_reward.append(running_reward)
				total_length.append(j)
				break

		
			#Update our running tally of scores.
		if i % 100 == 0:
			avg_rewards.append(np.mean(total_reward[-100:]))
			print(str((i/total_episodes)*100) + "%")
		i += 1
		
avgX = np.linspace(0, len(total_reward), len(avg_rewards))
#plt.plot(total_reward)
#plt.plot(avgX, avg_rewards)
plt.plot(avg_rewards)
plt.show()