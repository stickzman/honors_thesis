import numpy
import random 
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from thawedLakeEngine import Env

discountRate = .97
learnRate = .15
totalEps = 1000
		
#------------------------------------------------------------
#Implement the same learning algorithm from frozenlake.py

prevSuccess = False
firstSuccessEp = -1
totalSuccessEps = 0
lastFailedEp = -1
	
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
			if reward == env.GOAL_REWARD:
				if prevSuccess == False:
					#Record the first successful ep
					prevSuccess = True
					firstSuccessEp = e
				totalSuccessEps += 1
			else:
				lastFailedEp = e
			print("Episode finished after {} timesteps".format(t+1))
			break
			
print()
print("Percentage of successful episodes: " + str((totalSuccessEps/totalEps) * 100) + "%")
print()
print("First successful episode: " + str(firstSuccessEp))
print()
print("Last failed episode: " + str(lastFailedEp))