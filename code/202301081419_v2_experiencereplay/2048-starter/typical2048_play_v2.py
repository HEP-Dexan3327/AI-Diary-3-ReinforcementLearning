import gym
import gym_2048
from gym.utils import play
import pygame

# this version is just to play the game in human-controlled mode

env = gym.make('gym_2048/Typical2048', render_mode="rgb_array", size=4)

mapping = {
    (pygame.K_RIGHT,): 0, 
    (pygame.K_DOWN,): 1,
    (pygame.K_LEFT,): 2,
    (pygame.K_UP,): 3,
    (pygame.K_z,): 4,
}

play.play(env, keys_to_action=mapping, noop=None, fps=10)