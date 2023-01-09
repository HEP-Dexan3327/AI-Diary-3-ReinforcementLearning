import time
from collections import Counter

import gym
import pygame

import gym_2048
import sys

import matplotlib.pyplot as plt
import numpy as np
import random
from cmdargs import args

from typical2048_qlibrary import *

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D
from keras.optimizers import Adam

size = args.size
mode = args.mode
ws   = args.window_size

# for random agent, use play v1 script with -m human_rand
assert mode != "human_rand"

window_size = args.window_size

env = gym.make('gym_2048/Typical2048', render_mode=mode, size=size, window_size=window_size)

if args.fps is not None:
    env.metadata["render_fps"] = args.fps
else:
    env.metadata["render_fps"] = 1000000000

env.action_space.seed(args.seed)

# DQN model
model = tf.keras.models.load_model(args.file, compile=False)
print(model.summary())

config = model.get_config() # Returns pretty much every information about your model
shape = config["layers"][0]["config"]["batch_input_shape"]
print(f'Input shape: {shape}') # returns a tuple of width, height and channels

input_type = get_input_type(shape)

# Test the agent
test_episodes = args.episodes
max_steps = args.max_steps

episode = 0
total_score = 0
max_score = 0
high_tile = 0
total_steps = 0
min_steps_achieved = (2 << 15)
max_steps_achieved = 0

len_top_tiles = 10  # maximum number of games that we are keeping track of (the high score)

running = True
step = 0


def end():
    if episode > 0:
        print(f"Average score: {total_score / episode:.2f}\n" +
              f"Maximum score: {max_score:d}\n" + f"Highest tile: {2 ** high_tile:d}\n" +
              f"Average steps: {total_steps / episode:.2f} ([{min_steps_achieved} to {max_steps_achieved}])")

    env.close()


top_tiles = []

print("Training started ...")
for episode in range(test_episodes):
    state = env.reset(seed=args.seed)[0]  # [0] for observation only
    total_testing_rewards = 0

    info = {'available_dir': np.array([True, True, True, True]), 'score': 0, 'highTile': 0}

    for step in range(max_steps):
        if args.mode != 'rgb_array':
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    end()
                    exit()

        "Obtain Q-values from network."
        q_values = model(vectorize(state, info['available_dir'], type=input_type))

        "Select action based on q-value."
        action = choose(env, q_values, info['available_dir'])

        "Obtain q-value for the selected action."
        q_value = q_values[0, action]

        "Deterimine next state."
        new_state, reward, done, truncated, info = env.step(action)  # take action and get reward
        state = new_state

        # print(state, action)
        if done or truncated:
            if done:
                total_score += info['score']
                max_score = max(max_score, info['score'])
                high_tile = max(high_tile, info['highTile'])
                top_tiles += [2**info['highTile']]
                if len(top_tiles) > len_top_tiles:
                    del top_tiles[0]

                print(f"Episode {episode} succeeded in {step} steps with score {info['score']},"
                      f" high tile {2**info['highTile']}..., \n"
                      f"Highest tile frequencies: {top_tiles}"
                      f"\nq_values: {q_values}")

                total_steps += step
                max_steps_achieved = max(max_steps_achieved, step)
                min_steps_achieved = min(min_steps_achieved, step)
                break

end()
