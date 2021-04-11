import gym
env_name = "MountainCarContinuous-v0"
max_episodes = 1000
max_timesteps = 20 * 20 * 20  # 3 hierarchies
env = gym.make(env_name)

for episode_index in range(max_episodes):
    env.reset()
    num_timesteps_elapsed = 0
    total_reward = 0
    while True:
        a = env.action_space.sample()
        s_prime, r, done, info = env.step(a)
        total_reward += r
        num_timesteps_elapsed += 1
        if done or (num_timesteps_elapsed >= max_timesteps):
            break
    print(f'Episode: {episode_index:4} | Reward {total_reward:3}')
