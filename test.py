import gym 

env = gym.make("MountainCar-v0")

for i in range(10):
	print("-------------")
	s = env.reset()
	for t in range(50):
		s, r, d, _ = env.step(2)
		env.render()
		print(s)