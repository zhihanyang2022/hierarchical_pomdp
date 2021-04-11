from gym.envs.registration import register

register(
    id='continuous-mountain-car-v0',
    entry_point='gym_visual.envs:Continuous_MountainCarEnv',
)