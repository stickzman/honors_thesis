import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Indv:	
	def __init__(self, sess):
		self.sess = sess
		self.genome = []
		for var in tf.trainable_variables():
			size = sess.run(tf.size(var))#Get flattened size of variable
			self.genome.extend(np.random.rand(size))
		
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
			
	def buildFeedDict(self):
		feed_dict = {}
		i = 0
		for var in tf.trainable_variables():
			shape = sess.run(tf.shape(var))
			tensor = self.__buildTensor(shape, i)
			feed_dict[var] = tensor
		return feed_dict
		
		
tf.reset_default_graph()

i = tf.placeholder(shape=[2,3, 1],dtype=tf.float32)
h = slim.fully_connected(i, 5, biases_initializer=None)
out = slim.fully_connected(h, 1, activation_fn=None, biases_initializer=None)

init = tf.global_variables_initializer()		

sess = tf.Session()
agent = Indv(sess)

feed_dict = agent.buildFeedDict()
feed_dict[i] = [[[1], [2], [3]], [[4],[2],[1]]]
print(sess.run(out, feed_dict))