from pomdp_mountain_car import Continuous_MountainCar_Pomdp_Env
import gym


env = Continuous_MountainCar_Pomdp_Env()

rewards = []
for i in range(100):
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()