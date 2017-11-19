import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import matplotlib.pyplot as plt
from genAlgAgent import Population;


		
		
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



pop = Population('CartPole-v0', sess, state_in, output, genSize=25, numParents=15, genLength=5, numGens=10, minW=0, maxW=100, mutProb=0.001)
pop.run()
#bestAgent = pop.getBestAgent()
#bestAgent.viewRun()

plt.plot(pop.best)
plt.show()