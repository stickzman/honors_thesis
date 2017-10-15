import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

try:
	xrange = xrange
except:
	xrange = range

val_lr = .001
pol_lr = .001
gamma = 0.99

max_eps = 50000
max_timesteps = 201

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

#Define tensorflow graph
tf.reset_default_graph()

state_in = tf.placeholder(shape=[None,4], dtype=tf.float32)

#Define Critic

#Feed forward
hidden_vals_layer = slim.fully_connected(state_in, 8, biases_initializer=None)
value_output = slim.fully_connected(hidden_vals_layer, 1, biases_initializer=None, activation_fn=None)
#value_output = slim.fully_connected(state_in, 1, biases_initializer=None, activation_fn=None)

#Update Critic
discounted_rewards_holder = tf.placeholder(shape=[None],dtype=tf.float32)
val_loss = -tf.reduce_sum(tf.square(discounted_rewards_holder - value_output))
trainer = tf.train.AdamOptimizer(learning_rate=val_lr)
#trainer = tf.train.GradientDescentOptimizer(learning_rate=val_lr)
update_values = trainer.minimize(val_loss)



#Define Actor

#Feed-forward part of graph

hidden_pol_layer = slim.fully_connected(state_in, 8, biases_initializer=None)
output_pol = slim.fully_connected(hidden_pol_layer, 2, biases_initializer=None, activation_fn=tf.nn.softmax)

#Define update part of graph
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
advantage_holder = tf.placeholder(shape=[1], dtype=tf.float32)

#indexes = tf.range(0, tf.shape(output_pol)[0]) * tf.shape(output_pol)[1] + action_holder
#responsible_outputs = tf.gather(tf.reshape(output_pol, [-1]), indexes)
responsible_outputs_array = tf.slice(output_pol, [0, action_holder[0]], [tf.shape(output_pol)[0], 1])
responsible_outputs = tf.reshape(responsible_outputs_array, [-1])

pol_loss = -tf.reduce_mean(tf.log(responsible_outputs)*advantage_holder)

optimizer = tf.train.AdamOptimizer(learning_rate=pol_lr)
update_policy = optimizer.minimize(pol_loss)


init = tf.global_variables_initializer()

#Run episodes
env = gym.make('CartPole-v0')
total_rewards = []
avg_rewards = []

with tf.Session() as sess:
	sess.run(init)
	for epNum in range(max_eps):
		s = env.reset()
		running_reward = 0
		ep_history = []
		est_values = []
		for t in range(max_timesteps):
			a_dist = sess.run(output_pol, {state_in: [s]})
			action = np.random.choice(range(len(a_dist[0])), p=a_dist[0])
			
			last_state = s
			s, reward, done, _ = env.step(action)
			ep_history.append([last_state,action,reward,s])
			
			running_reward += reward
			#env.render()

			#Update policy
			if epNum > 0:
				new_state_value = sess.run(value_output, {state_in:[s]})[0]
				est_values.append(new_state_value[0])
				last_state_value = sess.run(value_output, {state_in:[last_state]})[0]
				advantage = new_state_value[0] - last_state_value[0]
				#print(last_state_value)
				sess.run(update_policy, {state_in:[last_state], action_holder: [action], advantage_holder: [advantage]})
			
			if done:
				ep_history = np.array(ep_history)
				ep_history[:,2] = discount_rewards(ep_history[:,2])

				#print("DiscRew:", ep_history[:,2])
				#print("EstVals:", est_values)
				#print("???????:", sess.run(value_output, {state_in:np.vstack(ep_history[:,0])}) )

				for i in range(0,len(ep_history)):
					sess.run(update_values, {state_in:[ep_history[i,0]], discounted_rewards_holder: [ep_history[i,2]]})
				#sess.run(update_values, {state_in:np.vstack(ep_history[:,0]), discounted_rewards_holder: ep_history[:,2]})
				
				total_rewards.append(running_reward)
				break
		
		if epNum % 100 == 0:
			avg_rewards.append(np.mean(total_rewards[-100:]))
			print(str((epNum/max_eps)*100) + "%")
			
avgX = np.linspace(0, len(total_rewards), len(avg_rewards))
plt.plot(total_rewards)
plt.plot(avgX, avg_rewards)
plt.show()