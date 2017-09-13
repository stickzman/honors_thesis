import numpy
import gym
import random
from gym import wrappers

y = .97
lr = .85
eps = 1000

def updateQMat(q, r, s, a, s_):
	r_ = max(q[s_][:])
	q[s][a] = q[s][a] + lr * (r + y * r_ - q[s][a])
	return



env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, '/tmp/recording', force=True)

qMatrix = numpy.zeros((env.observation_space.n, env.action_space.n))

for i in range(eps):
	observation = env.reset()
	for t in range(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')):
		action = numpy.argmax(qMatrix[observation][:] + numpy.random.randn(1, env.action_space.n)*(1/(i+1)))
		oldObservation = observation;
		observation, reward, done, info = env.step(action)
		updateQMat(qMatrix, reward, oldObservation, action, observation)
		env.render()
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break

env.close()
gym.upload('/tmp/recording', api_key='sk_fVhBRLT7S7e4MoHswIH5wg')