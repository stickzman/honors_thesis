import numpy as np
import random
from lakeEngineFullBoard import Env
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

def oneHotEncode(arr, size):
	'''
	res = []
	for type in arr.astype(np.int).tolist():
		onehot = np.zeros(size)
		onehot[type] = 1
		res.append(onehot)
	'''
	res = np.zeros(size)
	res[arr] = 1
	return res
	
lr = .01

tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
boardInput = tf.placeholder(shape=[1, 16],dtype=tf.float32)
playerInput = tf.placeholder(shape=[1, 16],dtype=tf.float32)
hidP = slim.fully_connected(playerInput, 20, biases_initializer=None)[0]
hidB = slim.fully_connected(boardInput, 20)[0]
Qout = slim.fully_connected([hidB, hidP], 4, activation_fn=None, biases_initializer=None)[0]
#W = tf.Variable(tf.random_uniform([1,5,4],0,0.01))
#Qout = tf.reduce_mean(tf.matmul(inputs1,W), 1)
predict = tf.argmax(Qout)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
updateQVals = trainer.minimize(loss)

init = tf.global_variables_initializer()



env = Env()

# Set learning parameters
y = .99
e = 1
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
		boardState, s = env.reset()
		s = oneHotEncode(s, 16)
		rAll = 0
		d = False
		j = 0
		#The Q-Network
		while j < 99:
			j+=1
			#Choose an action by greedily (with e chance of random action) from the Q-network
			a,allQ = sess.run([predict,Qout],feed_dict={playerInput:[s], boardInput:[boardState]})

			if np.random.rand(1) < e:
				a = random.randint(0, 3)
			#Get new state and reward from environment
			s1,r,d = env.step(a)
			s1 = oneHotEncode(s1, 16)
			#Obtain the Q' values by feeding the new state through our network
			Q1 = sess.run(Qout,feed_dict={playerInput:[s1], boardInput:[boardState]})
			#Obtain maxQ' and set our target value for chosen action.
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[a] = r + y*maxQ1
			#Train our network using target and predicted Q values
			sess.run([updateQVals],feed_dict={playerInput:[s], boardInput:[boardState], nextQ:[targetQ]})
			rAll += r
			s = s1
			#env.render()
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
				e = 1./((i/50) + 10)
				#e = 1/(i+1)
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
