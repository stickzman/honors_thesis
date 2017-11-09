import gym

env = gym.make('Pendulum-v0')

env.reset()

print(env.observation_space.sample())
'''
for i in range (50):
	env.step(0)
	env.render()
	'''