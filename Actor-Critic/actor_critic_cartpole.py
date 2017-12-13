import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

val_lr = 0.01
pol_lr = 0.001
discount_rate = .99
state_size = 4

max_eps = 5000
max_timesteps = 201


def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * discount_rate + r[t]
		discounted_r[t] = running_add
	return discounted_r

#Define Tensorflow graph
tf.reset_default_graph()

#Critic
state_in = tf.placeholder(shape=[None,state_size], dtype=tf.float32)
hidden_val_layer = slim.fully_connected(state_in, 4)
value_output = slim.fully_connected(hidden_val_layer, 1, biases_initializer=None, activation_fn=None)

#Actor
hidden_pol_layer = slim.fully_connected(state_in, 8, biases_initializer=None)
pol_output = slim.fully_connected(hidden_pol_layer, 2, biases_initializer=None, activation_fn=tf.nn.softmax)

#Update graph error

#Actor
advantage_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
indexes = tf.range(0, tf.shape(pol_output)[0]) * tf.shape(pol_output)[1] + action_holder
responsible_outputs = tf.gather(tf.reshape(pol_output, [-1]), indexes)
pol_loss = -tf.reduce_mean(tf.log(responsible_outputs)*advantage_holder)

pol_optimizer = tf.train.AdamOptimizer(learning_rate=pol_lr)
update_pol = pol_optimizer.minimize(pol_loss)

#Critic Update
value_holder = tf.placeholder(shape=[None, 1], dtype=tf.float32)
val_loss = tf.reduce_sum(tf.square(value_holder - value_output))
val_optimizer = tf.train.AdamOptimizer(learning_rate=val_lr)
update_values = val_optimizer.minimize(val_loss)


total_rewards = []
avg_rewards = []

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epNum in range(max_eps):
		s = env.reset()
		ep_rewards = 0
		ep_history = []
		for t in range(max_timesteps):
			a_dist = sess.run(pol_output, {state_in:[s]})
			a = np.random.choice(len(a_dist[0]), p=a_dist[0])
			next_state, r, d, _ = env.step(a)
			ep_history.append([s, r])
			#env.render()
			
			#Update Actor
			state_value = sess.run(value_output, {state_in:[s]})
			next_state_value = sess.run(value_output, {state_in:[next_state]})
			advantage = next_state_value[0] - state_value[0]
			sess.run(update_pol, {state_in:[s], action_holder:[a], advantage_holder:advantage})
			
			s = next_state
			ep_rewards += r
			if d:
				total_rewards.append(ep_rewards)
				if epNum%100 == 0:
					avg_rewards.append(np.mean(total_rewards[-100:]))
					print(str(epNum/max_eps*100) + "%")
				break
		#Update Critic		
		ep_history = np.array(ep_history)
		disc_rews = discount_rewards(ep_history[:,1])
		#Print the values predicted for each move this past episode
		print(sess.run(value_output, {state_in:np.vstack(ep_history[:,0])}))
		print("-------------------------------")
		#print(sess.run(val_loss, {value_holder:np.vstack(disc_rews), state_in:np.vstack(ep_history[:,0])}))
		sess.run(update_values, {value_holder:np.vstack(disc_rews), state_in:np.vstack(ep_history[:,0])})
		
avgX = np.linspace(0, len(total_rewards), len(avg_rewards))
plt.plot(total_rewards)
plt.plot(avgX, avg_rewards)
plt.show()