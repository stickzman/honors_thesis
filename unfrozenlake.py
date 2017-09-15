import numpy

discountRate = .97
learnRate = .85
totalEps = 1000

#-------------------------------------------------------------
#Recreate Frozen Lake w/o slipping function

class Env:
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
		self.lakeArray = numpy.zeros(16) #Initialize lake to all safe tiles
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
		reward = 0
		done = False
		newState = self.moveMatrix[self.pIndex][action]
		if newState == -1: 
			#If player attempted to move into a non-existent state,
			#do not move player and return
			return (self.pIndex, reward, done)
		self.pIndex = newState
		#Adjust the award according to current state
		if self.isWin():
			done = True
			reward = 1
		elif self.isFallen():
			done = True
			reward = -1
		
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
		
#------------------------------------------------------------
#Implement the same learning algorithm from frozenlake.py

prevSuccess = False
firstSuccessEp = -1
totalSuccessEps = 0
	
def updateQMat(q, reward, state, action, newState):
	futureReward = max(q[newState][:]) #Maximum future reward from new state
	q[state][action] = q[state][action] + learnRate * (reward + discountRate * futureReward - q[state][action])
	return

	
env = Env()
qMatrix = numpy.zeros((16, 4)) #Initialize qMatrix to 0s
for e in range(totalEps):
	state = env.reset()
	for t in range(1000):
		#Create an array of random estimated rewards representing each action
		#with the possible range of rewards decreasing with each episode
		randomActions = numpy.random.randn(1, 4)*(1/(e+1))
		#Choose either the action with max expected reward, or a random action
		#according to randomActions array. With each episode, the random actions
		#will become less chosen.
		action = numpy.argmax(qMatrix[state][:] + randomActions)
		oldObservation = state;
		state, reward, done = env.step(action)
		updateQMat(qMatrix, reward, oldObservation, action, state) #Update the Q-Matrix
		env.render()
		if done:
			if reward == 1:
				if prevSuccess == False:
					#Record the first successful ep
					prevSuccess = True
					firstSuccessEp = e
				totalSuccessEps += 1
			print("Episode finished after {} timesteps".format(t+1))
			break
			
print()
print("Percentage of successful episodes: " + str((totalSuccessEps/totalEps) * 100) + "%")
print()
print("First successful episode: " + str(firstSuccessEp))