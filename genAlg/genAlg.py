import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import matplotlib.pyplot as plt
from genAlgAgent import Population
from genAlgAgent14Genes import Population as Population14G
from genAlgAgent10Genes import Population as Population10G
import argparse

parser = argparse.ArgumentParser()
#Choose the number of genes to use with -g=[10, 14, or 60]
parser.add_argument("-g", "-geneType", type=int, default=60, choices=[10, 14, 60])
#Set the numpy random seed using -s
parser.add_argument("-s", "-seed", type=int, default=-1)
#Set agent to choose actions deterministically instead of stochastically
parser.add_argument("-d", "-deterministic", type=bool, default=False)
args = parser.parse_args()

populationSize=10
numParents=4
generationLength=20
numGenerations=10
minWeight=0
maxWeight=100
mutationProb=0.01


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

if args.s != -1:
	np.random.seed(args.s)

if args.g == 60:
	pop = Population('CartPole-v0', sess, state_in, output, genSize=populationSize, numParents=numParents, genLength=generationLength, numGens=numGenerations, minW=minWeight, maxW=maxWeight, mutProb=mutationProb, deterministic=args.d)
elif args.g == 14:
	pop = Population14G('CartPole-v0', sess, state_in, output, genSize=populationSize, numParents=numParents, genLength=generationLength, numGens=numGenerations, minW=minWeight, maxW=maxWeight, mutProb=mutationProb, deterministic=args.d)
elif args.g == 10:
	pop = Population10G('CartPole-v0', sess, state_in, output, genSize=populationSize, numParents=numParents, genLength=generationLength, numGens=numGenerations, minW=minWeight, maxW=maxWeight, mutProb=mutationProb, deterministic=args.d)
	
pop.run()
#bestAgent = pop.getBestAgent()
#bestAgent.viewRun()

plt.plot(pop.best)
plt.show()