import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from thawedLakeEngine import Env


lr = .1

tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
updateQVals = trainer.minimize(loss)

init = tf.global_variables_initializer()



env = Env()

# Set learning parameters
y = .99
e = 0.5
num_episodes = 1000

success = False
firstSuccessEp = -1
totalSuccessEps = 0
lastFailedEp = -1


#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_episodes):
		#Reset environment and get first new observation
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		#The Q-Network
		while j < 99:
			j+=1
			#Choose an action by greedily (with e chance of random action) from the Q-network
			a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
			print(a, allQ)
			if np.random.rand(1) < e:
				a[0] = random.randint(0, 3)
			#Get new state and reward from environment
			s1,r,d = env.step(a[0])
			#Obtain the Q' values by feeding the new state through our network
			Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
			#Obtain maxQ' and set our target value for chosen action.
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[0,a[0]] = r + y*maxQ1
			#Train our network using target and predicted Q values
			sess.run([updateQVals],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
			rAll += r
			s = s1
			env.render()
			if d == True:
				if r==1:
					totalSuccessEps += 1
					if success == False:
						success = True
						firstSuccessEp = i
				else:
					lastFailedEp = i
					print("FAILED")
				#Reduce chance of random action as we train the model.
				print("Episode finished after " + str(j) + " timesteps")
				#e = 1./((i/50) + 10)
				e = 1/(i+1)
				break
		rList.append(rAll)				
print()
print("Percent of successful episodes: " + str((totalSuccessEps/num_episodes)*100) + "%")
print()
print("First successful episode: " + str(firstSuccessEp))
print()
print("Last failed episode: " + str(lastFailedEp))


plt.plot(rList)

plt.show()
