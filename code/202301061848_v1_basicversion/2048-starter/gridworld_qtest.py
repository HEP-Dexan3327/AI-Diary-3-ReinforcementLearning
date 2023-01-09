import gym
import gym_boardgames
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from cmdargs import args

size = args.size
mode = args.mode

# for random agent, use play v1 script with -m human_rand
assert mode != "human_rand"  

env = gym.make('gym_2048/GridWorld-v0', render_mode=mode, size=size)

if args.fps is not None:
    env.metadata["render_fps"] = args.fps

env.action_space.seed(args.seed)

# Load the Q-table from file
filename = f"qtable_gridworld_{size}x{size}.txt"
if args.file is not None:
    filename = args.file
Q = np.loadtxt(filename)

# Test the agent
test_episodes = args.episodes
max_steps = args.max_steps


def encode(state):
    """Convert the state vector into a scalar for indexing the Q-table"""
    # e.g. for size = 5, there are 5 x 5 x 5 x 5 = 625 states
    # state [1, 2, 3, 4] => (1 * 5 + 2) * 5 + 3) * 5 + 4 = 194
    i = state[0]
    i *= size
    i += state[1]
    i *= size
    i += state[2]
    i *= size
    i += state[3]
    return i


print("Testing started ...")
success_episodes = 0
for episode in range(test_episodes):
    state = env.reset(seed=args.seed)[0]  # [0] for observation only
    state = encode(state)
    total_testing_rewards = 0
    for step in range(max_steps):
        action = np.argmax(Q[state, :]) 
        new_state, reward, done, _, info = env.step(action)  # take action and get reward
        state = new_state
        # print(state, action)
        state = encode(state)
        if done: # End the episode
            print(f"Episode {episode} succeeded in {step+1} steps with score {info['score']}...")
            success_episodes += 1
            break
    else:
        print(f"Episode {episode} truncated ...")

print(f"Success rate: {success_episodes/test_episodes:.2f}")

env.close()
