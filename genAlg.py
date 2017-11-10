import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym

class Indv:	
	def __init__(self, sess, env, maxW=1, minW=0):
		self.done = False
		self.input = env.reset()
		self.sess = sess
		self.genome = []
		for var in tf.trainable_variables():
			size = sess.run(tf.size(var))#Get flattened size of variable
			self.genome.extend((np.random.rand(size) * (maxW-minW)) + minW)
		self.weightDict = self.buildWeightDict()
			
	def step(self):
		if not self.done:
			feed_dict = self.weightDict.copy()
			feed_dict[state_in] = [self.input]
			a_dist = self.sess.run(output,feed_dict)
			#Stochastic Selection
			#a = np.random.choice(a_dist[0],p=a_dist[0])
			#a = np.argmax(a_dist == a)
			#Deterministic Selection
			a = np.argmax(a_dist)
			print(a_dist)
			self.input, r, self.done, _ = env.step(a)
			env.render()
		
	def __buildTensor(self, shape, i):
		if len(shape) == 1:
			tensor = []
			for idx in range(shape[0]):
				tensor.append(self.genome[i])
				i += 1
			return tensor
		else:
			tensor = []
			for idx in range(shape[0]):
				tensor.append(self.__buildTensor(shape[1:], i))
			return tensor
			
	def buildWeightDict(self):
		feed_dict = {}
		i = 0
		for var in tf.trainable_variables():
			shape = sess.run(tf.shape(var))
			tensor = self.__buildTensor(shape, i)
			feed_dict[var] = tensor
		return feed_dict
		
		
s_size = 4
h_size = 10
a_size = 2		

tf.reset_default_graph() #Clear the Tensorflow graph.
		
#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
state_in= tf.placeholder(shape=[1 , s_size],dtype=tf.float32)
hidden = slim.fully_connected(state_in, h_size, biases_initializer=None)
output = slim.fully_connected(hidden, a_size, biases_initializer=None, activation_fn=tf.nn.softmax)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

env = gym.make('CartPole-v0')
agent = Indv(sess, env, 5, -10)

while not agent.done:
	agent.step()