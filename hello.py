import gym
import gym_microrts
import time
from gym.wrappers import TimeLimit, Monitor


env = gym.make("MicrortsWorkerRush-v1")
#env = Monitor(env, f'videos')
env.action_space.seed(0)
obs = env.reset()
print(obs)
for i in range(10000):
    env.render()
    time.sleep(0.2)
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    if done:
        print("done")
        break
env.close()
