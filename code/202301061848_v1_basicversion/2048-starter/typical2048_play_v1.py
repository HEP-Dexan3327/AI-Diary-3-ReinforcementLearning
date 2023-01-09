import time

import gym
import gym_2048
import pygame
import sys
from cmdargs import args

"""
Running after doing 
        pip install -e gym-2048
Human player: run
        python 2048-starter/typical2048_play_v1.py -m 'human'
Random player (command line interface): run
        python 2048-starter/typical2048_play_v1.py -fps 100000000
Random player (with GUI): run
        python 2048-starter/typical2048_play_v1.py -m 'human_rand' -e 10 -fps 20
"""
render_mode = args.mode
if render_mode == "human_rand":
    render_mode = "human"

episodes = args.episodes
max_steps = args.max_steps
window_size = args.window_size

env = gym.make('gym_2048/Typical2048', render_mode=render_mode, size=args.size, window_size=window_size)
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

if args.fps is not None:
    env.metadata["render_fps"] = args.fps

env.action_space.seed(args.seed)
observation, info = env.reset(seed=args.seed)

episode = 0
total_score = 0
max_score = 0
high_tile = 0
total_steps = 0
min_steps_achieved = (2 << 15)
max_steps_achieved = 0

running = True
step = 0

while running and episode < episodes:
    action = None
    if args.mode != 'rgb_array':
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN and args.mode == "human":
                if event.key == pygame.K_RIGHT:
                    action = 0
                if event.key == pygame.K_DOWN:
                    action = 1
                if event.key == pygame.K_LEFT:
                    action = 2
                if event.key == pygame.K_UP:
                    action = 3
                if event.key == pygame.K_z:
                    action = 4
                if event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.QUIT:
                running = False
    if args.mode != "human":
        action = env.action_space.sample()  # random

    if action is not None:
        observation, reward, done, truncated, info = env.step(action)
        # print(episode, action, observation, reward, info)
        step += 1
        if done or truncated:
            if done:
                print(f"Episode {episode} succeeded in {step} steps with score {info['score']}...")
                total_score += info['score']
                max_score = max(max_score, info['score'])
                high_tile = max(high_tile, info['highTile'])

                total_steps += step
                max_steps_achieved = max(max_steps_achieved, step)
                min_steps_achieved = min(min_steps_achieved, step)
            else:
                print(f"Episode {episode} truncated ...")

            observation, info = env.reset(seed=args.seed)
            episode += 1
            step = 0

if episode > 0:
    print(f"Average score: {total_score / episode:.2f}\n" +
          f"Maximum score: {max_score:d}\n" + f"Highest tile: {2 ** high_tile:d}\n" +
          f"Average steps: {total_steps / episode:.2f} ([{min_steps_achieved} to {max_steps_achieved}])")

env.close()
