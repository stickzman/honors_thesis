import numpy as np
import random

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
	
	def __init__(self, slipRate = 0):
		self.SLIP_PERCENT = slipRate
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