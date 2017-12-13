import numpy as np
import random
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

learningRate = 0.02

env = gym.make("CartPole-v0")

tf.reset_default_graph()

#Build TensorFlow graph
observations = tf.placeholder(shape = [1,4], dtype=tf.float32)
weights = tf.Variable(tf.random_uniform([4,2],0,0.01))
Qvals = tf.matmul(observations, weights)
chosenAction = tf.argmax(Qvals, 1)

#Create loss function
realQvals = tf.placeholder(shape=[1, 2], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(realQvals - Qvals))
trainer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
update = trainer.minimize(loss)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


discountRate = 0.1
e = .5

totalEps = 5000

rList = []

with tf.Session() as sess:
	sess.run(init)
	for i in range(totalEps):
		rAll = 0
		first = False
		obs = env.reset()
		for t in range(200):
			action = sess.run(chosenAction, {observations: [obs]})[0]
			qVals = sess.run(Qvals, {observations: [obs]})
			#if np.random.rand(1) < e:
			#	action = random.randint(0, 1)
			newObs, reward, done, _ = env.step(action)
			#if done == True: reward = -1
			newQvals = sess.run(Qvals, {observations: [newObs]})
			futureReward = np.max(newQvals)
			qVals[0][action] = reward + discountRate * futureReward
			#Update the model
			sess.run(update, {realQvals: qVals, observations: [obs]})
			obs = newObs
			#if i%500 == 0: env.render()
			rAll += reward
			if done == True:
				e = 1/(i+1)
				#e = 1./((i/50) + 10)
				break
		rList.append(rAll)
		print("Completed episode " + str(i))

#Graph the total rewards per episode
plt.plot(rList)

plt.show()