import numpy as np
import gym
from policy_gradient_agent import Agent
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")

agent = Agent(.2, 16, 4, 20, 10, .99)

num_eps = 10000
total_rewards = []
avg_rewards = []
for i in range(num_eps):
	s = env.reset()
	running_reward = 0
	for t in range(999):
		a = agent.chooseAction(np.identity(16)[s:s+1][0])
		newS, r, d, _ = env.step(a)
		agent.observe(np.identity(16)[s:s+1][0], a, r, d)
		running_reward += r
		s = newS
		if d:
			if i%100 == 0:
				print(str(i/num_eps*100) + "%")
				avg_rewards.append(np.mean(total_rewards[-100:]))
			total_rewards.append(running_reward)
			break
			
avgX = np.linspace(0, len(total_rewards), len(avg_rewards))
plt.plot(total_rewards)
plt.plot(avgX, avg_rewards)
plt.show()