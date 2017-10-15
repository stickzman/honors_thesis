import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt
from tut_policy_gradient_agent import agent
from thawedLakeEngine import Env

try:
	xrange = xrange
except:
	xrange = range
	
env = Env()

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

myAgent = agent(lr=1e-2,s_size=1,a_size=4,h_size=10) #Load the agent.

total_episodes = 2000 #Set total number of episodes to train agent on.
max_ep = 201
update_frequency = 5

doTrain = True

init = tf.global_variables_initializer()

rList = []
avgList = []

saver = tf.train.Saver()

# Launch the tensorflow graph
with tf.Session() as sess:

	sess.run(init)
	
	print("Restore session?")
	restore = input("Y/N (No): ").lower()
	if len(restore) > 0 and restore[0] == 'y':
		saver.restore(sess, "tmp/model.ckpt")
		print("Model restored.")
		
		print("Continue training?")
		train = input("Y/N (Yes): ").lower()
		if len(train) > 0 and train[0] == 'n':
			doTrain = False
			print("Model will not be updated.")
		
		
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
			a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[[s]]})
			a = np.random.choice(a_dist[0],p=a_dist[0])
			a = np.argmax(a_dist == a)
			s1,r,d = env.step(a) #Get our reward for taking an action given a bandit.
			ep_history.append([s,a,r,s1])
			s = s1
			running_reward += r
			#if i%500==0: env.render()
			if d == True:
				#Update the network.
				if doTrain:
					ep_history = np.array(ep_history)
					ep_history[:,2] = discount_rewards(ep_history[:,2])
					
					feed_dict={myAgent.reward_holder:ep_history[:,2],
							myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
					grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
					for idx,grad in enumerate(grads):
						gradBuffer[idx] += grad
				
				
					if i % update_frequency == 0 and i != 0:
						feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
						_ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
					for ix,grad in enumerate(gradBuffer):
						gradBuffer[ix] = grad * 0
					
				total_reward.append(running_reward)
				total_length.append(j)
				rList.append(running_reward)
				break

		
			#Update our running tally of scores.
		if i % 100 == 0:
			avgList.append(np.mean(total_reward[-100:]))
			print(str((i/total_episodes)*100) + "%")
		
		#print(running_reward)
		i += 1

		
	avgX = np.linspace(0, len(rList), len(avgList))
	plt.plot(rList)
	plt.plot(avgX, avgList)
	plt.show()

	print("Save model?");
	save = input("Y/N (No): ").lower()
	if len(save) > 0 and save[0] == 'y':
		save_path = saver.save(sess, "tmp/model.ckpt")
		print("Model saved in file: %s" % save_path)

