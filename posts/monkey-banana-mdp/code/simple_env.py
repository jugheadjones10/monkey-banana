import numpy as np  # noqa
import pygame  # noqa

import gymnasium as gym  # noqa
from gymnasium import spaces  # noqa
from gymnasium.envs.registration import register  # noqa


class SimpleLineWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The length of the line
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's, chair's, and banana's locations.
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(1,), dtype=int),
                "banana": spaces.Box(0, size - 1, shape=(1,), dtype=int),
            }
        )

        # We have 7 actions: "left", "right", "grab banana"
        self.action_space = spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        """
        self._action_to_direction = {
            0: -1,  # "left"
            1: 1,  # "right"
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        # So that we can display curr action in the pygame interface
        self.curr_action = None

    def get_possible_actions(self, state):
        agent_location, banana_location = state
        if agent_location == 0:
            return [1, 2]
        elif agent_location == self.size - 1:
            return [0, 2]
        else:
            return [0, 1, 2]

    # Return all possible states subject to logical constraints
    def get_all_states(self):
        states = []
        for agent in range(self.size):
            for banana in range(self.size):
                states.append((agent, banana))
        return states

    def flatten_obs(self, obs):
        return (
            obs["agent"],
            obs["banana"],
        )

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "banana": self._banana_location,
        }

    def _get_info(self):
        return {"distance": abs(self._agent_location - self._banana_location)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize positions
        self._agent_location = self.np_random.integers(0, self.size, size=1, dtype=int)[
            0
        ]
        # self._banana_location = self.np_random.integers(
        #     0, self.size, size=1, dtype=int
        # )[0]

        # Use normal np random to init banana location
        self._banana_location = np.random.randint(0, self.size)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.curr_action = action
        terminated = False
        reward = -1

        # Move left or right
        if action in [0, 1]:
            direction = self._action_to_direction[action]
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        # Grab the banana
        elif action == 2 and (self._agent_location == self._banana_location):
            terminated = True
            reward = 10

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Render the current action in the top left corner
        font = pygame.font.Font(None, 36)
        # Map the action number to text
        if self.curr_action == 0:
            action_text = "left"
        elif self.curr_action == 1:
            action_text = "right"
        elif self.curr_action == 2:
            action_text = "grab banana"
        else:
            action_text = "unknown"
        text = font.render(action_text, True, (0, 0, 0))
        canvas.blit(text, (0, 0))

        # Draw the banana
        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                self._banana_location * pix_square_size,
                (self.size - 2) * pix_square_size,
                pix_square_size,
                pix_square_size,
            ),
        )
        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (
                (self._agent_location + 0.5) * pix_square_size,
                (self.size - 0.5) * pix_square_size,
            ),
            pix_square_size / 3,
        )

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
