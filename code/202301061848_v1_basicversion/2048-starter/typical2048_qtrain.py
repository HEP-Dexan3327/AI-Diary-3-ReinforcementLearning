import time
from collections import Counter

import gym
import pygame

import gym_2048
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import random
from cmdargs import args

from typical2048_qlibrary import *

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D
from keras.optimizers import Adam

import copy
import threading

from output import Output

size = args.size
mode = args.mode
ws   = args.window_size

save_interval = 50

learning_rate = .0005
lr_decay = 1
opt = Adam(learning_rate=learning_rate)
gamma = .95  # or .95
epsilon_decay = 1-1e-3
local_epsilon = [.5]*16

# for random agent, use play v1 script with -m human_rand
assert mode != "human_rand"

window_size = args.window_size

env = gym.make('gym_2048/Typical2048', render_mode=mode, size=size, window_size=window_size)

if args.fps is not None:
    env.metadata["render_fps"] = args.fps
else:
    env.metadata["render_fps"] = 1000000000

env.action_space.seed(args.seed)


#threading
num_threads = 1 if args.mode=='human' else 10
envs = [copy.deepcopy(env) for i in range(num_threads)]

buffer_capacity = None
best_capacity = None
input_file = args.file
ofile = args.output_file
if ofile is None:
    ofile = f"data/qtable_{time.strftime('%Y%m%d%H%M')}"
output = Output(ofile, 'model', output_every_n=save_interval)  # does any output job.
folder = output.dir

# DQN model
num_classes = 4
options_per_cell = 16  # 16 if onehot / all models on or before 202212110239
train_type = 'one-hot'  #'one-hot' if output 16
input_shape = (num_classes, size ** 2, options_per_cell)
epsilon_min = 0
epsilon = .5


if args.file is not None:
    model = tf.keras.models.load_model(args.file, compile=False)
    print(model.summary())
    train_type = get_input_type(model.shape)
else:
    "Dimensionality reduction by obtaining Q-value rows by using a neural network."
    model = Sequential(
        [
            tf.keras.Input(shape=input_shape),
            Reshape((num_classes, size, size, options_per_cell)),
            Conv2D(128, kernel_size=(2, 2), activation="relu", padding='SAME'),
            Conv2D(32, kernel_size=(2, 2), activation="relu", padding='SAME'),
            Reshape((-1,)),
            Dense(128, activation='relu'),
            Dense(num_classes)
        ]
    )
    model.build()


print(model.summary())
with open(f"{folder}/model_structure.txt", "a+") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

with open(f"{folder}/model_structure.txt", "a+") as f:
    f.write(f"Model trained from loaded file: {input_file}\n")
    f.write(f"Parameters: \tbuffer size: {buffer_capacity}\n")
    f.write(f"\t\t\t\t* best capacity: {best_capacity}\n")
    f.write(f"\t\t\t\t* gamma: {gamma}, epsilon: {epsilon} (decay = {epsilon_decay}, min = {epsilon_min})\n")
    f.write(f"\t\t\t\t* lr: {learning_rate} (decay = {lr_decay})\n")
    model.summary(print_fn=lambda x: f.write(x + "\n"))


def end():
    if episode > 0:
        print(f"Average score: {total_score / episode:.2f}\n" +
              f"Maximum score: {max_score:d}\n" + f"Highest tile: {2 ** high_tile:d}\n" +
              f"Average steps: {total_steps / episode:.2f} ([{min_steps_achieved} to {max_steps_achieved}])")

    env.close()


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

top_tiles = []

# threading
class TrainModel (threading.Thread):
    def __init__(self, env, episode):
        threading.Thread.__init__(self)
        self.env = env
        self.episode = episode
    def run(self):
        global max_score, high_tile, total_score, top_tiles, epsilon, \
            max_steps_achieved, min_steps_achieved, total_steps, output
        env = self.env
        episode = self.episode
        state = env.reset(seed=args.seed)[0]  # [0] for observation only
        total_testing_rewards = 0

        info = {'available_dir': np.array([True, True, True, True]), 'score': 0, 'highTile': 0}

        for step in range(max_steps):
            if args.mode != 'rgb_array':
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        output.output_model(episode, model)
                        end()
                        exit()

            with tf.GradientTape() as tape:  # tracing and computing the gradients ourselves.
                "Obtain Q-values from network."
                q_values = model(vectorize(state, info['available_dir'], type=train_type))

                "Select action using epsilon-greedy strategy."
                sample_epsilon = np.random.rand()
                thisgame_hi = info['highTile']
                self_epsilon = local_epsilon[thisgame_hi]
                if sample_epsilon <= self_epsilon:
                    action = env.action_space.sample()
                else:
                    action = choose(env, q_values, info['available_dir'])

                "Obtain q-value for the selected action."
                q_value = q_values[0, action]

                "Deterimine next state."
                new_state, reward, done, truncated, info = env.step(action)  # take action and get reward
                state = new_state

                "From the Q-learning update formula, we have:"
                "   Q'(S, A) = Q(S, A) + a * {R + λ argmax[a, Q(S', a)] - Q(S, A)}"
                "Target of Q' is given by: "
                "   R + λ argmax[a, Q(S', a)]"
                "Hence, MSE loss function is given by: "
                "   L(w) = E[(R + λ argmax[a, Q(S', a, w)] - Q(S, a, w))**2]"
                next_q_values = tf.stop_gradient(model(vectorize(new_state, info['available_dir'], type=train_type)))
                next_action = choose(env, next_q_values, info['available_dir'])
                next_q_value = next_q_values[0, next_action]

                observed_q_value = reward + (gamma * next_q_value)
                loss = (observed_q_value - q_value) ** 2

                def decay(ep: float)->float:
                    ep *= epsilon_decay
                    return max(ep, epsilon_min)

                self_epsilon = decay(self_epsilon)
                epsilon = decay(epsilon)
                for i in range(thisgame_hi):
                    local_epsilon[i] = min(decay(local_epsilon[i]), decay(local_epsilon[thisgame_hi]))

                "Computing and applying gradients"
                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

                # print(state, action)
                if done or truncated:
                    if done:
                        total_score += info['score']
                        max_score = max(max_score, info['score'])
                        high_tile = max(high_tile, info['highTile'])
                        top_tiles += [2 ** info['highTile']]
                        if len(top_tiles) > len_top_tiles:
                            del top_tiles[0]

                        soutput = f"Episode {episode} succeeded in {step} steps with score {info['score']}," \
                                 f" high tile {2 ** info['highTile']}..., \n" \
                                 f"Highest tile frequencies: {top_tiles}" \
                                 f"\nepsilon: {self_epsilon}; q_values: {q_values}"
                        print(soutput)

                        with open(f"{folder}/_descriptions.txt", "a+") as f:
                            f.write(soutput + "\n")
                        with open(f"{folder}/_data.txt", "a+") as f:
                            f.write(f"{episode}\t{step}\t{info['score']}\t{info['highTile']}\n")

                        output.log(done, episode, step, info, model=model, do_output=False, epsilon=self_epsilon)

                        total_steps += step
                        max_steps_achieved = max(max_steps_achieved, step)
                        min_steps_achieved = min(min_steps_achieved, step)
                        break
    def join(self):
        threading.Thread.join(self)

save_interval = 50

print("Training started ...")
trainThreads = []
for episode in range(test_episodes):
    thread = TrainModel(envs[episode % num_threads], episode)
    thread.start()
    trainThreads.append(thread)
    if episode % num_threads == num_threads-1:
        [trainThread.join() for trainThread in trainThreads]
        trainThreads = []
    if episode % save_interval > save_interval - 5:
        output.concat({'episode': [episode], 'best': [np.NAN], 'good': [np.NAN]})
        output.output_img(episode=-1)


end()
