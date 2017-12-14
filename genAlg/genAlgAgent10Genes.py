import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import matplotlib.pyplot as plt
import hashlib
from collections import Iterable

def flatten(lis):
		for item in lis:
			if isinstance(item, Iterable):
				for x in flatten(item):
					yield x
			else:        
				yield item


class Population:	
	def __init__(self, envName, sess, state_in, output, genSize, numParents, numGens, genLength=5, minW=0, maxW=1, crsProb=.90, mutProb=.0001, deterministic=False):
		self.best = []
		self.genLength = genLength
		self.minW = minW
		self.maxW = maxW
		self.mutProb = mutProb
		self.numGens = numGens
		self.crossoverProb = crsProb
		self.numParents = numParents//2*2
		self.gen = []
		self.oldGenIDs = []
		self.sess = sess
		for i in range(genSize):
			self.gen.append(Indv(sess, state_in, output, gym.make(envName), minW=minW, maxW=maxW, deterministic=deterministic))
			
	def run(self):
		for i in range(self.numGens):
			self.runGen()
			print("-------Gen " + str(i) + "-------")
			self.displayGen()
			self.displayNew()
			self.oldGenIDs = self.__getIDs(self.gen)
			parents = self.__parentSelection()			
			self.__crossover(parents)
			for agent in self.gen:
				if agent.done:
					#if agent not in parents:
						#agent.updateGenome()
					agent.reset()
	
	def runGen(self):
		for t in range(self.genLength):
			while self.__minOneAgentRunning():
				for agent in self.gen:
					agent.step()
			self.__next()
		fit = self.__getFitness()
		self.best.append(np.amax(fit))
		print("Best fitness: " + str(self.best[-1]))
		
	def displayGen(self):
		gen = []
		fitIDs = np.argsort(self.__getFitness())
		for id in fitIDs:
			gen.append(self.gen[id])
		for agent in gen:
			print("ID: " + str(agent.id), "Fitness: " + str(agent.avgFitness()))
			
	def displayNew(self):
		oldGen = set(self.oldGenIDs)
		gen = set(self.__getIDs(self.gen))
		newIDs = gen - oldGen
		newAgents = []
		for id in newIDs:
			#Finds agent that has a matching id and adds it to the newAgents list
			newAgents.append(next(agent for agent in self.gen if agent.id == id))
		print("--------New Agents--------")
		for agent in newAgents:
			print("ID: " + str(agent.id), "Fitness: " + str(agent.avgFitness()))
		
	def getBestAgent(self):
		fit = self.__getFitness()
		return self.gen[np.argsort(fit)[-1]]
	
	def __getIDs(self, gen):
		ids = []
		for agent in gen:
			ids.append(agent.id)
		return ids
	
	def __next(self):
		for agent in self.gen:
			agent.next()
		
	def __parentSelection(self):
		'''
		probs = self.__getProbs()
		parentProbs = np.random.choice(probs, self.numParents, False, probs)
		parents = []
		for parent in parentProbs:
			parents.append(self.gen[np.argmax(probs == parent)])
		'''
		fitness = self.__getFitness()
		idxs = np.argsort(fitness)[-self.numParents:]
		parents = []
		for id in idxs:
			parents.append(self.gen[id])
		return parents
		
	def __crossover(self, parents):
		offspringGenes = []
		for i in range(0, len(parents), 2):
			p1 = parents[i]
			p2 = parents[i+1]
			g1 = p1.genome
			g2 = p2.genome
			kid1 = []
			kid2 = []
			for gene1, gene2 in zip(g1, g2):
				if np.random.rand() > .5:
					kid1.append(gene1)
					kid2.append(gene2)
				else:
					kid1.append(gene2)
					kid2.append(gene1)
			kid1 = self.__mutate(kid1)
			kid2 = self.__mutate(kid2)
			offspringGenes.append(kid1)
			offspringGenes.append(kid2)
		weakIdxs = np.argsort(self.__getFitness())[:len(offspringGenes)]
		for i, gene in zip(weakIdxs, offspringGenes):
			self.gen[i].reset(gene)
			
	def __mutate(self, genome):
		for gene in genome:
			if np.random.rand() < self.mutProb:
				#print("MUTATION")
				gene = (np.random.rand(len(gene)) * (self.maxW-self.minW)) + self.minW
		return genome
		
		
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
			fitness.append(agent.avgFitness())
		return fitness
		
	def __minOneAgentRunning(self):
		res = False
		for indv in self.gen:
			if indv.done == False: res = True
		return res



class Indv:	
	def __init__(self, sess, state_in, output, env, minW=0, maxW=1, deterministic=False, genome=None):
		self.state_in = state_in
		self.output = output
		self.env = env
		self.sess = sess
		self.maxW = maxW
		self.minW = minW
		self.determ = deterministic
		self.updateGenome(genome)
		self.reset()
			
	def step(self, render=False):
		if not self.done:
			feed_dict = self.weightDict.copy()
			feed_dict[self.state_in] = [self.input]
			a_dist = self.sess.run(self.output,feed_dict)
			
			if (self.determ):
				#Deterministic Selection
				a = np.argmax(a_dist)
			else:
				#Stochastic Selection
				a = np.random.choice(a_dist[0],p=a_dist[0])
				a = np.argmax(a_dist == a)
			
			self.input, r, self.done, _ = self.env.step(a)
			self.fitness += r
			if render: self.env.render()
	
	def viewRun(self):
		self.reset()
		i = 0
		while not self.done:
			i+= 1
			self.step(render = True)
		print(i)
		
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
			
	def __buildWeightDict(self):
		feed_dict = {}
		vars = tf.trainable_variables()
		tensor1 = []
		tensor2 = []
		for gene in self.genome:
			tensor1.append(gene[0].tolist())
			tensor2.append(gene[1])
		tensor1 = self.sess.run(tf.reshape(tensor1, [4,10]))
		feed_dict[vars[0]] = tensor1
		feed_dict[vars[1]] = tensor2
		self.weightDict = feed_dict
		
	def __genGenome(self):
		self.genome = []
		vars = tf.trainable_variables()
		hsize = self.sess.run(tf.shape(vars[0]))[1]#Get size of hidden layer
		lsize = self.sess.run(tf.shape(vars[0]))[0]#Size of left side
		rsize = self.sess.run(tf.shape(vars[1]))[1]#Size of right side
		for i in range(hsize):
			gene = []
			gene.append((np.random.rand(lsize) * (self.maxW-self.minW)) + self.minW)
			gene.append((np.random.rand(rsize) * (self.maxW-self.minW)) + self.minW)
			self.genome.append(gene)
		
	def reset(self, genome=None):
		if genome!=None: self.updateGenome(genome)
		self.done = False
		self.input = self.env.reset()
		self.fitness = 0
		self.totalFits = []
		
	def next(self):
		self.done = False
		self.input = self.env.reset()
		self.totalFits.append(self.fitness)
		self.fitness = 0
		
	def avgFitness(self):
		return np.average(self.totalFits)
	
	def updateGenome(self, genome=None):
		if genome==None:
			self.__genGenome()
		else:
			self.genome = genome
		self.id = hash(tuple(list(flatten(self.genome))))
		self.__buildWeightDict()