import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Agent():	
	def __init__(self, lr, s_size, a_size, h_size, b_size, gamma):
		self.exp_buffer = []
		self.end_of_last_ep = 0
		self.ep_count = 0
		self.b_size = b_size
		self.gamma = gamma
		
		tf.reset_default_graph() #Clear the Tensorflow graph.
		
		#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
		self.state_in= tf.placeholder(shape=[None, s_size],dtype=tf.float32)
		hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None)
		self.output = slim.fully_connected(hidden, a_size, biases_initializer=None, activation_fn=tf.nn.softmax)

		#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
		#to compute the loss, and use it to update the network.
		self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
		self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
		
		self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
		self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
		
		self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
		
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		self.min_loss = optimizer.minimize(self.loss)
		
		init = tf.global_variables_initializer()
		
		self.sess = tf.Session()
		self.sess.run(init)
		
	def __discount_rewards(self, r):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
		running_add = 0
		for t in reversed(range(0, r.size)):
			running_add = running_add * self.gamma + r[t]
			discounted_r[t] = running_add
		return discounted_r
		
	def __update(self):
		'''
		discounted_history = []
		for ep_history in batch_hist:
			ep_history = np.array(ep_history)
			ep_history[:, 2] = self.discount_rewards(ep_history[:, 2], gamma)
			for t in ep_history:
				discounted_history.append(t)
		discounted_history = np.array(discounted_history)
		'''
		feed_dict={self.reward_holder:self.exp_buffer[:,2],
			self.action_holder:self.exp_buffer[:,1],self.state_in:np.vstack(self.exp_buffer[:,0])}
		self.sess.run(self.min_loss, feed_dict)
		self.exp_buffer = []
		self.ep_count = 0
		
	def chooseAction(self, s):
		a_dist = self.sess.run(self.output,feed_dict={self.state_in:[s]})
		a = np.random.choice(a_dist[0],p=a_dist[0])
		return np.argmax(a_dist == a)
		
	def observe(self, s, a, r, d):
		self.exp_buffer.append([s, a, r])
		if d:
			self.ep_count += 1
			self.exp_buffer = np.array(self.exp_buffer)
			self.exp_buffer[self.end_of_last_ep:, 2] = self.__discount_rewards(self.exp_buffer[self.end_of_last_ep:, 2])
			self.end_of_last_ep = len(self.exp_buffer) - 1
			if self.ep_count >= self.b_size:
				self.__update()
			else:
				self.exp_buffer = self.exp_buffer.tolist()
			