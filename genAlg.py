import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym

class Population:
	def __init__(self, envName, sess, genSize, numParents, numGens, minW=0, maxW=1, crossoverProb=.90):
		self.numGens = numGens
		self.crossoverProb = crossoverProb
		self.numParents = numParents//2*2
		self.gen = []
		self.sess = sess
		for i in range(genSize):
			self.gen.append(Indv(sess, gym.make(envName), minW=minW, maxW=maxW))
			
	def run(self):
		for i in range(self.numGens):
			self.runGen()
			parents = self.__parentSelection()
			self.__crossover(parents)
			for agent in self.gen:
				if agent.done:
					if agent not in parents: agent.updateGenome()
					agent.reset()
		
			
	def runGen(self):
		while self.__minOneAgentRunning():
			for agent in self.gen:
				agent.step()
		fit = self.__getFitness()
		print("Avg fitness: " + str(np.average(fit)),  "Best fitness: " + str(np.amax(fit)))
		
	def __parentSelection(self):
		probs = self.__getProbs()
		parentProbs = np.random.choice(probs, self.numParents, False, probs)
		parents = []
		for parent in parentProbs:
			parents.append(self.gen[np.argmax(probs == parent)])
		return parents
		
	def __crossover(self, parents):
		offspringGenes = []
		for i in range(0, len(parents), 2):
			p1 = parents[i]
			p2 = parents[i+1]
			g1 = p1.genome
			g2 = p2.genome
			newG = []
			for gene1, gene2 in zip(g1, g2):
				if np.random.rand() > .5:
					newG.append(gene1)
				else:
					newG.append(gene2)
			offspringGenes.append(newG)
		weakIdxs = np.argsort(self.__getFitness)[:len(offspringGenes)]
		for weakId, gene in zip(weakIdxs, offspringGenes):
			self.gen[i].reset(gene)
			
			
		
	def __getProbs(self):
		fitness = self.__getFitness()
		total = sum(fitness)
		probs = []
		for fit in fitness:
			probs.append(fit/total)
		return probs
	
	def __getFitness(self):
		fitness = []
		for agent in self.gen:
			fitness.append(agent.fitness)
		return fitness
		
	def __minOneAgentRunning(self):
		res = False
		for indv in self.gen:
			if indv.done == False: res = True
		return res



class Indv:	
	def __init__(self, sess, env, minW=0, maxW=1, genome=None):
		self.env = env
		self.sess = sess
		self.maxW = maxW
		self.minW = minW
		self.__genGenome()
		self.reset()
		self.weightDict = self.buildWeightDict()
			
	def step(self):
		if not self.done:
			feed_dict = self.weightDict.copy()
			feed_dict[state_in] = [self.input]
			a_dist = self.sess.run(output,feed_dict)
			
			#Stochastic Selection
			a = np.random.choice(a_dist[0],p=a_dist[0])
			a = np.argmax(a_dist == a)
			
			#Deterministic Selection
			#a = np.argmax(a_dist)
			
			self.input, r, self.done, _ = self.env.step(a)
			self.fitness += r
			#self.env.render()
		
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
		
	def __genGenome(self):
		self.genome = []
		for var in tf.trainable_variables():
			size = sess.run(tf.size(var))#Get flattened size of variable
			self.genome.extend((np.random.rand(size) * (self.maxW-self.minW)) + self.minW)
		
	def reset(self, genome=None):
		if genome!=None: self.updateGenome(genome)
		self.done = False
		self.input = self.env.reset()
		self.fitness = 0
		
	def updateGenome(self, genome=None):
		if genome==None:
			self.__genGenome()
		else:
			self.genome = genome
		
		
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


pop = Population('CartPole-v0', sess, genSize=50, numParents=25, numGens=40, maxW=10)
pop.run()