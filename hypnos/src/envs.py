""" There are several things going on here:
    
    1. ALE - a fully featured atari_py wrapper with the configuration of the Rainbow paper.
    2. Dopamine wrapper of the Atari in the OpenAI Gym.
    3. A Gym-like API for MinAtar.
    4. Various PyTorch wrappers and utility functions.
"""
import random
from collections import deque

import atari_py
import cv2
import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete
from minatar import Environment

try:
    import cv2
except ModuleNotFoundError as err:
    print(
        "OpenCV is required when using the ALE env wrapper. ",
        "Try `conda install -c menpo opencv`.",
        err,
    )

__all__ = [
    "get_env",
    "get_wrapped_atari",
    "TorchWrapper",
]


def get_env(opt, mode="train", **kwargs):
    """ Configures an environment based on the name and options. """
    if opt.game.split("-")[0] in ("LunarLander", "CartPole"):
        return TorchWrapper(gym.make(opt.game), opt.device)
    if opt.game.split("-")[0] == "MinAtar":
        return TorchWrapper(MinAtar(opt.game.split("-")[-1], **kwargs), opt.device)
    # probably an ALE game
    if "stochasticity" not in kwargs:
        kwargs["stochasticity"] = opt.stochasticity
    return get_wrapped_atari(opt.game, device=opt.device, mode=mode, **kwargs)


def get_wrapped_atari(
    env_name, mode="train", dopamine=False, stochasticity="random_starts", **kwargs,
):
    """ The preprocessing traditionally used by DeepMind on Atari.
    """
    hist_len = kwargs.get("hist_len", 4)
    seed = kwargs.get("seed", np.random.randint(100_000))
    if stochasticity == "random_starts":
        random_starts, sticky_action_p = True, 0
    elif stochasticity == "sticky_actions":
        random_starts, sticky_action_p = False, 0.25
    else:
        raise ValueError(f"{stochasticity} stochasticity scheme not available.")

    if dopamine:
        print(
            "WARNING: Dopamine wrapper does not,"
            + " provide a DONE signal at loss of life."
        )
        if stochasticity == "random_starts":
            raise NotImplementedError("Dopamine doesn't do random_starts.")
        return dopamine_env(env_name.capitalize(), sticky_actions=True)
    return ALE(
        env_name,
        seed,
        kwargs.get("device", torch.device("cpu")),
        random_starts=random_starts,
        sticky_action_p=sticky_action_p,
        history_length=hist_len,
        training=(mode == "train"),
    )


class ALE:
    """ A wrapper over atari_py, the Arcade Learning Environment python
    bindings that provides: frame concatentation of `history_len`, sticky
    actions probability, end game after first life in training mode, clip
    rewards during training.
    
    All credits for this wrapper go to
    [@Kaixhin](https://github.com/Kaixhin/Rainbow/blob/master/env.py)
    
    Returns:
        env: An ALE object with settings close to the original DQN paper.
    """

    # pylint: disable=too-many-arguments, bad-continuation
    def __init__(
        self,
        game,
        seed,
        device,
        training=True,
        clip_rewards_val=1,
        history_length=4,
        random_starts=True,
        sticky_action_p=0,
        max_episode_length=108e3,
    ):
        # pylint: enable=bad-continuation
        self.game_name = game
        self.device = device

        # Manage stochasticity
        self.sticky_action_p = sticky_action_p
        self.random_starts = random_starts
        if (sticky_action_p != 0) and random_starts:
            raise ValueError("Can't use both random_starts and sticky actions.")
        if (sticky_action_p == 0) and not random_starts:
            print("ALE: WARNING, the game is deterministic!")

        # Configure ALE
        self.ale = atari_py.ALEInterface()
        self.ale.setInt("random_seed", seed)
        self.ale.setInt("max_num_frames_per_episode", max_episode_length)
        self.ale.setFloat("repeat_action_probability", sticky_action_p)
        self.ale.setInt("frame_skip", 0)
        self.ale.setBool("color_averaging", False)
        self.ale.loadROM(
            atari_py.get_game_path(game)
        )  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.action_space = Discrete(len(self.actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        # Used to check if resetting only from loss of life
        self.life_termination = False
        self.window = history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=history_length)
        self.training = training  # Consistent with model training mode
        self.clip_val = clip_rewards_val

    def _get_state(self):
        state = cv2.resize(
            self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_AREA,
        )
        return torch.tensor(state, dtype=torch.uint8, device=self.device)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(
                torch.zeros(84, 84, device=self.device, dtype=torch.uint8)
            )

    def _random_noops(self):
        # Perform up to 30 random no-ops before starting
        for _ in range(random.randrange(30)):
            self.ale.act(0)  # Assumes raw action 0 is always no-op
            if self.ale.game_over():
                self.ale.reset_game()

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Do the random ops
            if self.random_starts:
                self._random_noops()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device, dtype=torch.uint8)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                # Only set flag when not truly done
                self.life_termination = not done
                done = True
            self.lives = lives
        # clip the reward
        if self.clip_val and self.training:
            reward = max(min(reward, self.clip_val), -self.clip_val)
        # Return state, reward, done
        state = torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()
        return state, reward, done, {}

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def render(self):
        frame = self.ale.getScreenRGB()[:, :, ::-1]
        frame = cv2.resize(frame, (320, 420), interpolation=cv2.INTER_LANCZOS4)
        cv2.imshow("screen", frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def __str__(self):
        mode = "train" if self.training else "eval"
        if self.random_starts:
            stochasticity = "no_op_30"
        elif self.sticky_action_p != 0:
            stochasticity = f"{self.sticky_action_p:.2f}_sticky_action"
        else:
            stochasticity = "deterministic"

        return "ALE(game={}, mode={}, stochasticity={}, hist_len={}, repeat_act=4)".format(
            self.game_name, mode, stochasticity, self.window
        )


class MinAtar(gym.Env):
    """ Minatar does not have a Gym-like API so we make one.
    """

    def __init__(self, game, **kwargs):
        self._env = env = Environment(game, **kwargs)
        self._action_space = Discrete(env.num_actions())
        self._observation_space = Box(
            low=False, high=True, shape=env.state_shape(), dtype=np.bool
        )

    def reset(self):
        self._env.reset()
        return self._env.state()

    def step(self, action):
        reward, done = self._env.act(action)
        observation_ = self._env.state()
        return observation_, reward, done, {}

    def render(self):
        raise NotImplementedError("Better fails, right?")

    def close(self):
        pass

    @property
    def action_space(self):
        """ Return the action space """
        return self._action_space

    @property
    def observation_space(self):
        """ Return the observation space """
        return self._observation_space

    @property
    def sticky_action_prob(self):
        return self._env.sticky_action_prob

    @sticky_action_prob.setter
    def sticky_action_prob(self, value):
        self._env.sticky_action_prob = value


class TorchWrapper(gym.ObservationWrapper):
    """ Receives numpy arrays and returns torch tensors.
        Used with robotics envs.
    """

    def __init__(self, env, device):
        super().__init__(env)
        self._device = device

    def observation(self, obs):
        obs = torch.from_numpy(obs)
        if obs.ndim == 3:
            obs = obs.permute(2, 0, 1).byte().to(self._device)
            return obs.view(1, 1, *obs.shape)
        return obs.view(1, 1, -1).float().to(self._device)


def dopamine_env(game_name=None, sticky_actions=True):
    """ Wraps an Atari 2600 Gym environment with some basic preprocessing.

        This preprocessing matches the guidelines proposed in Machado et al.
        (2017), "Revisiting the Arcade Learning Environment: Evaluation
        Protocols and Open Problems for General Agents".

        The created environment is the Gym wrapper around the Arcade Learning
        Environment.

        The main choice available to the user is whether to use sticky
        actions or not. Sticky actions, as prescribed by Machado et al.,
        cause actions to persist with some probability (0.25) when a new
        command is sent to the ALE. This can be viewed as introducing a mild
        form of stochasticity in the environment. We use them by default.

        Args:
            game_name: str, the name of the Atari 2600 domain.
            sticky_actions: bool, whether to use sticky_actions as per Machado et al.

        Returns:
            An Atari 2600 environment with some standard preprocessing.
  """
    assert game_name is not None
    game_version = "v0" if sticky_actions else "v4"
    full_game_name = "{}NoFrameskip-{}".format(game_name, game_version)
    env = gym.make(full_game_name)
    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames.
    # We handle this time limit internally instead, which lets us cap at 108k
    # frames (30 minutes). The TimeLimit wrapper also plays poorly with saving
    # and restoring states.
    env = env.env
    env = AtariPreprocessing(env)
    return env


class AtariPreprocessing(object):
    """ A class implementing image preprocessing for Atari 2600 agents.

        Specifically, this provides the following subset from the JAIR paper
        (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

            * Frame skipping (defaults to 4).
            * Terminal signal when a life is lost (off by default).
            * Grayscale and max-pooling of the last two frames.
            * Downsample the screen to a square image (defaults to 84x84).

        More generally, this class follows the preprocessing guidelines set down in
        Machado et al. (2018), "Revisiting the Arcade Learning Environment:
        Evaluation Protocols and Open Problems for General Agents".
    """

    def __init__(
        self, environment, frame_skip=4, terminal_on_life_loss=False, screen_size=84,
    ):
        """ Constructor for an Atari 2600 preprocessor.

            Args:
            environment: Gym environment whose observations are preprocessed.
            frame_skip: int, the frequency at which the agent experiences the game.
            terminal_on_life_loss: bool, If True, the step() method returns
                is_terminal=True whenever a life is lost. See Mnih et al. 2015.
            screen_size: int, size of a resized Atari 2600 frame.

            Raises:
            ValueError: if frame_skip or screen_size are not strictly positive.
        """
        if frame_skip <= 0:
            raise ValueError(
                "Frame skip should be strictly positive, got {}".format(frame_skip)
            )
        if screen_size <= 0:
            raise ValueError(
                "Target screen size should be strictly positive, got {}".format(
                    screen_size
                )
            )

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        obs_dims = self.environment.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        ]

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        return Box(
            low=0,
            high=255,
            shape=(self.screen_size, self.screen_size, 1),
            dtype=np.uint8,
        )

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def reset(self):
        """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
        self.environment.reset()
        self.lives = self.environment.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def render(self, mode):
        """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
        return self.environment.render(mode)

    def step(self, action):
        """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
        accumulated_reward = 0.0

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, info = self.environment.step(action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self.environment.ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            if is_terminal:
                break
            # We max-pool over the last two frames, in grayscale.
            elif time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

        # Pool the last two observations.
        observation = self._pool_and_resize()

        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
        self.environment.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(
                self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0],
            )

        transformed_image = cv2.resize(
            self.screen_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)


if __name__ == "__main__":
    pass
