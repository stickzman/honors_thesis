import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import matplotlib.pyplot as plt
import sys
from genAlgAgent import Population
from genAlgAgent14Genes import Population as Population14G
from genAlgAgent10Genes import Population as Population10G

genSize=25
numParents=10
genLength=20
numGens=10
minW=0
maxW=100
mutProb=0.01


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

if len(sys.argv) > 2:
	seed = int(sys.argv[2])
	np.random.seed(seed)

if len(sys.argv) <= 1 or sys.argv[1] == "0":
	pop = Population('CartPole-v0', sess, state_in, output, genSize=genSize, numParents=numParents, genLength=genLength, numGens=numGens, minW=minW, maxW=maxW, mutProb=mutProb)
elif sys.argv[1] == "14":
	pop = Population14G('CartPole-v0', sess, state_in, output, genSize=genSize, numParents=numParents, genLength=genLength, numGens=numGens, minW=minW, maxW=maxW, mutProb=mutProb)
elif sys.argv[1] == "10":
	pop = Population10G('CartPole-v0', sess, state_in, output, genSize=genSize, numParents=numParents, genLength=genLength, numGens=numGens, minW=minW, maxW=maxW, mutProb=mutProb)
	
pop.run()
#bestAgent = pop.getBestAgent()
#bestAgent.viewRun()

plt.plot(pop.best)
plt.show()