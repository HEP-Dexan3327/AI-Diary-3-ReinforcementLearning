import collections

import numpy as np

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    """
    Reference: https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
    There is not much that I could improve the code. @credit: Jordi TORRES.AI
    """
    def __init__(self, capacity, best_capacity=256):
        self.buffer = collections.deque(maxlen=capacity)
        self.best = []
        self.age = 0
        self.best_capacity = best_capacity

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.age += 1
        self.buffer.append(experience)
        # based on the concept that high reward should be prioritized
        # we would make sure that older experiences with high reward values is not dumped
        self.best += [(self.age, experience)]
        self.best = sorted(self.best, key=lambda x: x[1][2]+x[0]/10000, reverse=True)
        if len(self.best) > self.best_capacity:
            self.best.pop()
        # i.e. reward + age/1000

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states \
            = zip(*([self.buffer[idx] for idx in indices]
                    + [x[1] for x in self.best]))

        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)