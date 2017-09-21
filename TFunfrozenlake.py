import numpy as np
import random
import tensorflow as tf

#-------------------------------------------------------------
#Recreate Frozen Lake w/o slipping function

class Env:
	GOAL_REWARD = 1
	HOLE_REWARD = -1
	DEFAULT_REWARD = 0
	
	SLIP_PERCENT = 0

	#Representation of tiles in lake array
	SAFE_TILE = 0
	HOLE_TILE = 1
	START_TILE = 3
	GOAL_TILE = 4
	
	#Representation of moves in move array
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	
	def __init__(self):
		self.pIndex = 0 #Player index, also the current state
		self.lakeArray = np.zeros(16) #Initialize lake to all safe tiles
		self.lakeArray[0] = self.START_TILE
		self.lakeArray[15] = self.GOAL_TILE
		self.lakeArray[5] = self.HOLE_TILE
		self.lakeArray[7] = self.HOLE_TILE
		self.lakeArray[11] = self.HOLE_TILE
		self.lakeArray[12] = self.HOLE_TILE
		
		#Initialize move matrix to allow movement anywhere
		self.moveMatrix = []
		for s in range(16):
			self.moveMatrix.append([s-4, s+4, s-1, s+1])
		
		#Restrict movement for edge tiles
		for i in range(4):
			self.moveMatrix[i][self.UP] = -1
		for i in range(0, 13, 4):
			self.moveMatrix[i][self.LEFT] = -1
		for i in range(3, 16, 4):
			self.moveMatrix[i][self.RIGHT] = -1
		for i in range(12, 16):
			self.moveMatrix[i][self.DOWN] = -1
	
	#Check if the player is on the Goal tile
	def isWin(self):
		return self.pIndex == 15
		
	#Check if player is in a hole
	def isFallen(self):
		return self.pIndex == 5 or self.pIndex == 7 or self.pIndex == 11 or self.pIndex == 12
	
	#Execute the action and advance one timestep
	#Return the state, reward, and if the episode is done
	def step(self, action):
		reward = self.DEFAULT_REWARD
		done = False
		rnd = random.random()
		sliprate = self.SLIP_PERCENT
		if rnd < sliprate:
			randomMove = random.randint(0, 3)
			newState = self.moveMatrix[self.pIndex][randomMove]
		else:
			newState = self.moveMatrix[self.pIndex][action]
		if newState == -1: 
			#If player attempted to move into a non-existent state,
			#do not move player and return
			return (self.pIndex, reward, done)
		self.pIndex = newState
		#Adjust the award according to current state
		if self.isWin():
			done = True
			reward = self.GOAL_REWARD
		elif self.isFallen():
			done = True
			reward = self.HOLE_REWARD
		
		return (self.pIndex, reward, done)

	#Display the current environment	
	def render(self):
		s = ""
		for i in range(len(self.lakeArray)):
			if i%4 == 0:
				s += "\n"
			if i == self.pIndex:
				s += "P"
			elif self.lakeArray[i] == self.SAFE_TILE:
				s += "-"
			elif self.lakeArray[i] == self.HOLE_TILE:
				s += "O"
			elif self.lakeArray[i] == self.START_TILE:
				s += "S"
			elif self.lakeArray[i] == self.GOAL_TILE:
				s += "G"
		
		print(s)

	#Reset the environment
	def reset(self):
		self.pIndex = 0
		return self.pIndex
		


tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

env = Env()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000

success = False
firstSuccessEp = -1
totalSuccessEps = 0
lastFailedEp = -1


#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_episodes):
		#Reset environment and get first new observation
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		#The Q-Network
		while j < 99:
			j+=1
			#Choose an action by greedily (with e chance of random action) from the Q-network
			a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
			if np.random.rand(1) < e:
				a[0] = random.randint(0, 3)
			#Get new state and reward from environment
			s1,r,d = env.step(a[0])
			#Obtain the Q' values by feeding the new state through our network
			Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
			#Obtain maxQ' and set our target value for chosen action.
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[0,a[0]] = r + y*maxQ1
			#Train our network using target and predicted Q values
			_,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
			rAll += r
			s = s1
			env.render()
			if d == True:
				if r==1:
					totalSuccessEps += 1
					if success == False:
						success = True
						firstSuccessEp = i
				else:
					lastFailedEp = i
					print("FAILED")
				#Reduce chance of random action as we train the model.
				print("Episode finished after " + str(j) + " timesteps")
				#e = 1./((i/50) + 10)
				e = 1/(i+1)
				break
print()
print("Percent of successful episodes: " + str((totalSuccessEps/num_episodes)*100) + "%")
print()
print("First successful episode: " + str(firstSuccessEp))
print()
print("Last failed episode: " + str(lastFailedEp))
