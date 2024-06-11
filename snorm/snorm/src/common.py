# This code is from openai baseline
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env

from copy import deepcopy
from multiprocessing import Pipe, Process

import gym
import jax
import jax.numpy as jnp
import numpy as np
import rlog
import torch
from brax import envs
from termcolor import colored as clr

import src.centipede_env  # pylint: disable=unused-import

MUJOCO_ENVS = [
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    "HumanoidStandup-v2",
    "InvertedDoublePendulum-v2",
    "InvertedPendulum-v2",
    "Reacher-v2",
    "Swimmer-v2",
    "Walker2d-v2",
    "CentipedeFour-v2",
    "CentipedeSix-v2",
    "CentipedeEight-v2",
    "CentipedeTen-v2",
    "CentipedeTwelve-v2",
]
BRAX_ENVS = ["ant", "humanoid", "fetch", "grasp", "halfcheetah", "ur5e", "reacher"]
BOX2D_ENVS = ["LunarLanderContinuous-v2", "LunarLander-v2"]


def get_env(name="ant", env_no=1, device="cuda", normalize_obs=False):
    if normalize_obs and env_no == 1:
        rlog.info(
            clr(
                "Normalized observations are not supported in the validation env.",
                "red",
            )
        )
    assert (
        name in MUJOCO_ENVS + BRAX_ENVS + BOX2D_ENVS
    ), "env.name not in any of the available envs."
    if name in BRAX_ENVS:
        # env = JaxToTorchWrapper(
        #     create_gym_env(env_name, batch_size=batch_size, episode_length=1000),
        #     device=device,
        # )
        env = JaxToTorchWrapper(
            envs.create(name, batch_size=env_no, episode_length=1000),
            device=device,
            normalize_obs=normalize_obs,
        )
    elif name in BOX2D_ENVS or name in MUJOCO_ENVS:

        def make_env():
            def _thunk():
                env = gym.make(name)
                return env

            return _thunk

        if env_no == 1:
            env = ActionWrapper(TorchWrapper(gym.make(name), device=device))
        else:
            env = SubprocVecEnv(
                [make_env() for i in range(env_no)],
                device=device,
                normalize_obs=normalize_obs,
            )
    return env


class TorchWrapper(gym.ObservationWrapper):
    """ Applies a couple of transformations depending on the mode.
        Receives numpy arrays and returns torch tensors.
    """

    def __init__(self, env, device):
        super().__init__(env)
        self._device = device

    def observation(self, obs):
        return torch.from_numpy(obs).float().unsqueeze(0).to(self._device)

    def __len__(self):
        return 1


class ActionWrapper(gym.ActionWrapper):
    """ Torch to Gym-compatible actions.
    """

    def __init__(self, env):
        super().__init__(env)
        self._action_type = "Z" if hasattr(env.action_space, "n") else "R"

    def action(self, action):
        if self._action_type == "Z":
            if torch.is_tensor(action):
                return action.cpu().item()
            else:
                return action
        return action.flatten().cpu().numpy()

    def __len__(self):
        return 1


try:
    import cupy

    def _jax2torch(x, device="cuda"):
        """ Convert observations, reward and done signals from DeviceArray to tensors.
            Importantly, this happens without moving data from the device.
            Due to a PyTorch issue we need to pipe this via cupy.
            https://github.com/pytorch/pytorch/issues/32868
        """
        return torch.as_tensor(cupy.asarray(x), device=device)


except ImportError:  # sometimes there's no GPU in your life

    def _jax2torch(x, device="cpu"):
        """ Convert observations, reward and done signals from DeviceArray to tensors.
        """
        return torch.as_tensor(np.asarray(x), device=device)


def _torch2jax(x):
    """ Convert actions from torch to jax. Jax does not support imports yet so
        we need to move data from device to CPU.
        https://github.com/google/jax/issues/6222
    """
    return jnp.array(x.cpu().numpy())


def wrap(core_env):
    """ Detect when an episode in the batch terminates and reset it.
        Returns a wrapped state and step function for training.
    """

    def step(s0, state, action):
        state = core_env.step(state, action)

        def test_done(a, b):
            if a is s0.done:
                return b
            test_shape = [a.shape[0],] + [1 for _ in range(len(a.shape) - 1)]
            return jnp.where(jnp.reshape(state.done, test_shape), a, b)

        state = jax.tree_multimap(test_done, s0, state)
        return state

    return jax.jit(step)


class RunningMeanStd:
    def __init__(self, shape, device, epsilon=1e-4) -> None:
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.shape = shape

    def update(self, x):
        batch_cnt = x.shape[0]
        batch_mean = torch.mean(x, axis=0)
        batch_var = torch.var(x, axis=0)

        delta = self.mean - batch_mean
        tot_cnt = self.count + batch_cnt

        self.mean += delta * batch_cnt / tot_cnt
        m_a = self.var * self.count
        m_b = batch_var * batch_cnt
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_cnt / tot_cnt
        self.var = M2 / tot_cnt
        self.count = tot_cnt


def apply_normalization(x, stats, update_stats):
    assert x.shape == stats.shape, f"Shapes do not match, {x.shape}!={stats.shape}."
    if update_stats:
        stats.update(x)  # update the running mean and var
    x -= stats.mean  # center data
    x /= torch.sqrt(stats.var + 1e-08)  # standardize
    x.clip_(-10.0, 10.0)
    return x


class JaxToTorchWrapper:
    def __init__(self, env, seed=0, device="cuda", normalize_obs=False) -> None:
        self._env = env
        self.key = jax.random.PRNGKey(seed)
        self.device = device
        self.normalize_obs = normalize_obs

        # set the dimensions
        env_no = self._env.batch_size
        obs_size = self._env.observation_size
        obs_high = (np.inf * np.ones((env_no, obs_size))).astype(np.float32)
        action_high = np.ones((env_no, self._env.action_size), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

        if self.normalize_obs:
            self.obs_running_stats = RunningMeanStd(
                self.observation_space.shape, device
            )
            self.is_training = True

        # important bits
        def reset(key):
            """ When this is called we create an extra key that we use as
                seed for the next reset so that resets are not identical.
            """
            rng = jax.random.split(key, self._env.batch_size + 1)
            state = self._env.reset(rng[1:])
            return state, rng[0]

        self._reset = jax.jit(reset)
        self._step = None
        self._s0, self._state = None, None

    def reset(self):
        self._state, self.key = self._reset(self.key)
        self._s0 = deepcopy(self._state)  # we use this to detect ending episodes
        if self._step is None:
            # this is expensive so we only do it once.
            self._step = wrap(self._env)
        obs = _jax2torch(self._state.obs, device=self.device)
        if self.normalize_obs:
            obs = apply_normalization(obs, self.obs_running_stats, self.is_training)
        return obs

    def step(self, action):
        device = self.device
        self._state = s = self._step(self._s0, self._state, _torch2jax(action))
        o, r, d = [_jax2torch(x, device=device) for x in (s.obs, s.reward, s.done)]
        if self.normalize_obs:
            o = apply_normalization(o, self.obs_running_stats, self.is_training)
        return o, r, d, {}

    def seed(self, seed):
        self.key = jax.random.PRNGKey(seed)

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def __len__(self):
        return self._env.batch_size

    def __repr__(self) -> str:
        return f"JaxToTorchWrapper(VectorizedBraxEnv, norm_obs={self.normalize_obs})"


# class JaxToTorchWrapper(gym.Wrapper):
#     """Wrapper that converts Jax tensors to PyTorch tensors."""

#     def __init__(self, env, device=None):
#         """ Creates a Wrapper around a `GymWrapper` or `VectorGymWrapper` that
#             outputs PyTorch tensors.
#         """
#         super().__init__(env)
#         self.device = device

#     def reset(self):
#         obs = super().reset()
#         return self._observation(obs)

#     def step(self, action):
#         action = self._action(action)
#         obs, reward, done, info = super().step(action)
#         obs = self._observation(obs)
#         reward = self._reward(reward)
#         done = self._done(done)
#         info = self._info(info)
#         return obs, reward, done, info

#     def _observation(self, observation):
#         return _jax2torch(observation, device=self.device)

#     def _action(self, action):
#         return _torch2jax(action)

#     def _reward(self, reward):
#         return _jax2torch(reward, device=self.device)

#     def _done(self, done):
#         return _jax2torch(done, device=self.device)

#     def _info(self, info):
#         return {k: _jax2torch(v, device=self.device) for k,v in info.items()}


# ======================================
# Vectorized Env Wrapper using Subprocs
# ======================================


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == "reset":
            ob = env.reset()
            remote.send(ob)
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "close":
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, device="cuda", normalize_obs=False):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        self.device = device
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.normalize_obs = normalize_obs
        if self.normalize_obs:
            self.obs_running_stats = RunningMeanStd(
                (nenvs, observation_space.shape[-1]), device
            )
            self.is_training = True

    def step_async(self, actions):
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions = actions.flatten()
        for remote, action in zip(self.remotes, actions.cpu().numpy()):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        obs, rews, dones = [
            torch.from_numpy(x).float().to(self.device)
            for x in (np.stack(obs), np.stack(rews), np.stack(dones))
        ]
        if self.normalize_obs:
            obs = apply_normalization(obs, self.obs_running_stats, self.is_training)
        return obs, rews, dones, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = (
            torch.from_numpy(np.stack([remote.recv() for remote in self.remotes]))
            .float()
            .to(self.device)
        )
        if self.normalize_obs:
            return apply_normalization(obs, self.obs_running_stats, self.is_training)
        return obs

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return torch.from_numpy(
            np.stack([remote.recv() for remote in self.remotes])
        ).to(self.device)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
            self.closed = True

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def __len__(self):
        return self.nenvs

    def __str__(self) -> str:
        return "SubProcVecEnv(N={}, norm_obs={})".format(self.nenvs, self.normalize_obs)
