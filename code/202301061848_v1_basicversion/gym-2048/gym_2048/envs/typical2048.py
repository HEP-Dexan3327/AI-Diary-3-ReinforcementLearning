import time

import gym
from gym import spaces
import pygame
import numpy as np


"Arrows Key to play, Ctrl + Z to undo. You can only undo once consecutively."


class Typical2048Env(gym.Env):
    metadata = {"render_modes": ["ai", "human", "rgb_array"], "render_fps": 20, "window_size": 16}

    def __init__(self, render_mode=None, size=4, window_size=16):
        self._grid = None
        self._last_grid = None
        self._merged = None
        self._epoch = 0

        self.size = size  # The size of the square grid
        self.ws = window_size
        self.window_size = 512 * window_size / 16  # The size of the PyGame window
        self.bar_size = 100 * window_size / 16
        self.bar = np.array((0, self.bar_size))

        self.prob = (.9, .1)  # (.9, .1)
        self.action = -1

        self.reward_list_length = 1
        self.score = 0
        self.undo_score = 0
        self.reward = 0
        self.rewards = [0,]
        self.max_score = 0
        self.undo_unused = True
        self.punishment = -50

        self.available_dir = np.array([True, True, True, True])

        # Observations are 16-element lists, storing the numbers at each cell.
        # There are 16 possible numbers, ranging from 2**1 to 2**16.
        # The id 0 is reserved for EMPTY cell.
        self.observation_space = spaces.Box(0, size * size, shape=(size * size,), dtype=int)

        # We have 4 actions, corresponding to "right", "down", "left", "up" and "undo"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "down" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 1]),  # right  1st axis
            1: np.array([0, 1]),  # down   0th axis
            2: np.array([1, -1]),  # left   1st axis
            3: np.array([0, -1]),  # up     0th axis
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.ai = render_mode == 'ai'
        if self.ai:
            self.render_mode = 'human'

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._grid

    def _get_info(self):
        return {
            "highTile": max(self._grid),
            "score": self.score,
            "available_dir": self.available_dir
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # self.np_random.integers
        if self._epoch != 0:
            if self.render_mode == 'human':
                # The following line copies our drawings from `canvas` to the visible window
                s = pygame.Surface((self.window_size, self.window_size+self.bar_size))
                s.set_alpha(196)
                s.fill((64, 64, 64))
                pygame.display.update(self.window.blit(s, (0,0)))
                self._print_text(self.window, f'Game Over!', (self.window_size/2,self.window_size/2),
                                 color=(255, 255, 255))
                self._print_text(self.window, f'Your Score is {self.score}.',
                                 (self.window_size/2,self.window_size/2+100), color=(255, 255, 255))
                self.window.blit(self.window, self.window.get_rect())
                pygame.event.pump()
                pygame.display.update()
                if not self.ai:
                    time.sleep(5)

                # We need to ensure that human-rendering occurs at the predefined framerate.
                # The following line will automatically add a delay to keep the framerate stable.
                self.clock.tick(self.metadata["render_fps"])
        self._epoch += 1

        self.max_score = max(self.max_score, self.score)

        self.punishment = -50
        self.action = -1
        self.score = 0
        self.reward = 0
        self.undo_score = 0
        self.rewards = [0,]
        self.available_dir = np.array([True, True, True, True])
        self.undo_unused = True
        # Spawn the grid with 2 random tiles (2 or 4, i.e. code = 1 or 2)
        self._grid = np.random.permutation((0,) * (self.size ** 2 - 2) +
                                           tuple(np.random.choice((1, 2), size=(2,), p=self.prob)))
        self._last_grid = self._grid.copy()
        self._merged = np.zeros((self.size**2,))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _move_row(self, p: np.ndarray, m: np.ndarray, direction: int, do_reward=False) -> tuple[bool, np.ndarray]:
        """
        :param p: an ndarray row representing the numbers in a row (or column)
        :param m: an ndarray row representing whether the number is JUST merged.
        :param direction: an integer signifying whether to slide in the POSITIVE (+1) or NEGATIVE (-1) direction.
        :param do_reward: an bool signifying if we update the reward value.
        """
        out = np.zeros_like(p)
        mout = np.zeros_like(m)
        last = 0
        lastidx = 0
        direction *= -1
        # if direction is -1, i.e., towards the LEFT, then we DON'T need to reverse the array.
        # similarly, we NEED to reverse the row if direction is 1.

        idx = -1
        for i, e in enumerate(p[::direction]):
            if e == 0:
                continue
            if e != last or m[i] != 0 or m[lastidx] != 0:  # either not equal, or one of them is used.
                idx += 1
                out[idx] = e
                mout[idx] = m[i]
                last = int(e)
            else:
                out[idx] = last+1   # merge tiles
                mout[idx] = 1
                if do_reward:
                    self.reward += 2 ** (last + 1)
                    self.score += 2 ** (last + 1)
                last = 0
        m[:] = mout
        return not np.all(p-out[::direction] == 0), out[::direction]

    def _move_tiles(self, grid: np.ndarray, merge_grid: np.ndarray, action=0, do_reward=False) -> bool:
        g = grid.reshape((self.size, self.size))
        mg = merge_grid.reshape((self.size, self.size))
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # direction[0]: axis.
        # direction[1]: +/- value for that axis.
        OUT = False
        for i in range(self.size):
            p = g[:, i] if direction[0] else g[i, :]
            m = mg[:, i] if direction[0] else mg[i, :]
            out, p[:] = self._move_row(p, m, direction[1], do_reward=do_reward)
            if out:
                OUT = True
                if not do_reward:
                    return True

        grid[:] = g.reshape((-1,))
        return OUT

    def is_full(self) -> bool:
        ar = [self._move_tiles(self._grid.copy(), self._merged.copy(), action=action) for action in range(4)]
        self.available_dir = np.array(ar)# + [self.undo_unused])
        return not (True in self.available_dir)

    def _spawn_tile(self):
        empty_tiles = self._grid[self._grid == 0]
        self._grid[self._grid == 0] = np.random.permutation((0,)*(len(empty_tiles)-1) +
                                                            tuple(np.random.choice((1, 2), size=(1,), p=self.prob)))

    def step(self, action):
        self.reward = 0
        self.action = action
        if action == 4:
            if self.undo_unused:
                self._grid[:] = self._last_grid.copy()
                self.score = self.undo_score
                self.undo_unused = False
                self.reward = self.punishment / 10
            else:
                self.reward = self.punishment
                self.punishment -= 50
        elif action is not None:
            self._last_grid = self._grid.copy()
            self.undo_score = self.score
            self.undo_unused = True
            if not self.is_full():
                moved = 0
                while self._move_tiles(self._grid, self._merged, action=action, do_reward=True):
                    moved = 1
                self._merged = np.zeros_like(self._merged)
                if moved:
                    self._spawn_tile()
                else:
                    self.reward = 0
        # An episode is done iff the agent has reached the target
        terminated = self.is_full()
        if terminated:
            self.reward -= 10
        self.rewards += [self.reward]
        if len(self.rewards) > self.reward_list_length:
            self.rewards = self.rewards[1:]
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, self.reward, terminated, False, info

    def _get_color(self, tile: int) -> tuple[int, int, int]:
        colors = (
            (237, 228, 218),  # 2,      1
            (236, 223, 199),  # 4,      2
            (243, 177, 121),  # 8,      3
            (245, 149, 99),   # 16,     4
            (245, 124, 97),   # 32,     5
            (237, 87, 55),    # 64,     6
            (236, 206, 113),  # 128,    7
            (237, 204, 98),   # 256,    8
            (236, 199, 80),   # 512,    9
            (236, 197, 64),   # 1024,   10
            (236, 197, 1),    # 2048,   11
            (94, 220, 151),   # 4096,   12
            (236, 77, 88),    # 8192,   13
            (37, 186, 99),    # 16384,  14
            (0, 124, 189),    # 32768,  15
            (0, 0, 0)         # 65536,  16
        )
        return colors[tile-1]

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _debug_text(self, canvas: pygame.Surface, text: str, pos=(300, 300), color=(150, 100, 100)):
        self.__print_some_text(self.debug_font, canvas, text, pos, color)

    def _print_text(self, canvas: pygame.Surface, text: str, pos=(300, 300), color=(100, 100, 100)):
        self.__print_some_text(self.font, canvas, text, pos, color)

    def __print_some_text(self, font, canvas: pygame.Surface, text: str, pos=(300, 300), color=(100, 100, 100)):
        text_surf = font.render(text, False, color)
        text_rect = text_surf.get_rect()
        text_rect.center = pos
        canvas.blit(text_surf, text_rect)

    def _render_frame(self):
        if self.window is None and self.render_mode in ("human", "rgb_array"):
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size+self.bar_size))
            self.font = pygame.font.SysFont('Garamond', 50 * self.ws // 16)
            self.debug_font = pygame.font.SysFont('Garamond', 20 * self.ws // 16)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size+self.bar_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels
        margin = pix_square_size * .1

        # Drawing the tiles.
        for i in range(self.size):
            for j in range(self.size):
                tile = self._grid[i*self.size + j]
                if tile != 0:
                    rect = pygame.Rect(
                            self.bar + pix_square_size * np.array((i, j)) + margin,
                            (pix_square_size - margin * 2, pix_square_size - margin * 2),
                        )
                    pygame.draw.rect(
                        canvas,
                        self._get_color(tile),
                        rect,
                    )
                    self._print_text(canvas, str(2 ** tile), pos=rect.center)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, self.bar_size+pix_square_size * x),
                (self.window_size, self.bar_size+pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, self.bar_size),
                (pix_square_size * x, self.window_size+self.bar_size),
                width=3,
            )

        self._print_text(canvas, f'Score: {self.score}   Hi: {self.max_score}',
                         pos=(self.window_size/2, self.bar_size/2))
        self._debug_text(canvas, f'Action: {self.action}', pos=(self.window_size/2, self.bar_size/4))
        self._debug_text(canvas, f'Available: {self.available_dir}', pos=(self.window_size / 2, self.bar_size *3 / 4))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
