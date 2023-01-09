from gym.envs.registration import register

register(
    id="gym_2048/Typical2048",
    entry_point="gym_2048.envs:Typical2048Env",
)
