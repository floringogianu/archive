""" Hope for the best. There are a bunch of tricks in this PPO, although I tried to
    keep them as few as possibile:
    1. The two networks are initialized with orthogonal weights.
    2. The initial value of the scale is around 0.5. We do this by substracting
        a constant before the transformation: `softplus(logits - c)`.
    3. The advantages are standardized by default.
"""
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial

import gym
import rlog
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as O
from liftoff import parse_opts
from termcolor import colored as clr
from torch import nn

import src.io_utils as ioutil
from src.common import get_env, CloudpickleWrapper
from src.estimators import hook_spectral_normalization
from multiprocessing.context import SpawnContext
import torch.multiprocessing as mp


class TanhNormal:
    """ A Gaussian policy transformed by a Tanh function."""

    def __init__(self, action_num, min_std=0.0005, init_std=0) -> None:
        self.action_num = action_num
        self.min_std = min_std
        # find a constant such that the initial scale to be close to 0.5
        # c = softplus^-1(wanted_std - min_std)
        self.c_rho = torch.tensor(0.5 - min_std).expm1().log() if init_std else 0.0

    def __call__(self, x):

        loc, scale = torch.split(x, self.action_num, -1)
        if self.c_rho:
            scale = F.softplus(scale - self.c_rho) + self.min_std
        else:
            scale = F.softplus(scale) + self.min_std

        normal = D.Normal(loc, scale)
        pi = D.TransformedDistribution(
            normal, [D.transforms.TanhTransform(cache_size=1)]
        )
        return pi, normal.entropy()

    def __repr__(self) -> str:
        return "tanh(Normal(loc: {a}, scale: {a}))".format(a=("B", self.action_num))


class Discrete:
    """A discrete policy."""

    def __init__(self, action_num) -> None:
        self.action_num = action_num

    def __call__(self, x):
        x = x.unsqueeze(1)  # so that we get samples of shape (B, 1)
        pi = D.Categorical(logits=x)
        return pi, pi.entropy()

    def __repr__(self) -> str:
        return "Categorical(logits: {a})".format(a=("B", 1, self.action_num))


def _make_mlp(dims):
    layers = []
    for i in range(1, len(dims)):
        layers.append(nn.Linear(dims[i - 1], dims[i]))
        if i != (len(dims) - 1):
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """ A parametrized ActorCritic policy that can be either discrete or continuous. """

    def __init__(self, num_inputs, actor_outdim, policy, layer_dims=None, **kwargs):
        super(ActorCritic, self).__init__()

        self.policy = policy

        self.actor = _make_mlp([num_inputs, *layer_dims, actor_outdim])
        self.critic = _make_mlp([num_inputs, *layer_dims, 1])

        self.w0 = kwargs.get("init", "kaiming_uniform")
        self.reset_parameters()

        spectral = kwargs.get("spectral", None)
        if spectral:
            self.hooked_actor = hook_spectral_normalization(
                spectral["actor"], self.actor.named_children()
            )
            self.hooked_critic = hook_spectral_normalization(
                spectral["critic"], self.critic.named_children()
            )

    def forward(self, x):
        value = self.critic(x)
        pi_out = self.actor(x)
        pi, entropy = self.policy(pi_out)
        return pi, value, entropy

    def reset_parameters(self):
        def _orthogonal_init(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data, gain=1.41)
                torch.zero_(m.bias.data)

        if self.w0 == "orthogonal":
            rlog.info(clr(f"Reseting with {self.w0} init.", "green"))
            self.actor.apply(_orthogonal_init)
            self.critic.apply(_orthogonal_init)
            # we also scale the weights of the last layer of the policy, dunno why
            last_pi_layer = list(self.actor.children())[-1]
            nn.init.orthogonal_(last_pi_layer.weight.data, gain=0.01)
            # we also scale the weights of the last layer of the policy, dunno why
            last_val_layer = list(self.critic.children())[-1]
            nn.init.orthogonal_(last_val_layer.weight.data, gain=1.0)

    def __str__(self):
        return super().__str__() + "\nPolicy={} | w0={}".format(self.policy, self.w0)


Result = namedtuple("Result", ["avgR", "N", "full"])


class ParallelAverager:
    """ Something that can keep an average across multiple envs. It only uses values
        from complete episodes.
    """

    def __init__(self, B, device, name=None) -> None:
        self.name = name or None
        self.cnt = 0
        self.cumsums = torch.zeros((B,), device=device)
        self.buffer = torch.zeros((B,), device=device)
        self.Ns = torch.zeros((B,), device=device)

    def put(self, value, N):
        """ Add a list of values to a cumsum and keep track of the denominator
            used when averaging. A buffer is used so that we only track data from
            complete episodes.
        """
        self.buffer += value
        if N.any():
            to_avg = N == 1  # create a mask of series that can be averaged
            self.cumsums[to_avg] += self.buffer[to_avg]
            self.buffer[to_avg] = 0  # reset the buffer
            self.Ns += N
        self.cnt += 1

    def stats(self):
        """ Returns the stats."""
        idxs = self.Ns.nonzero()
        cumsums = self.cumsums[idxs]
        Ns = self.Ns[idxs]

        return Result(
            avgR=(cumsums / Ns).mean().item(), N=self.Ns.sum(), full=cumsums / Ns,
        )

    def reset(self):
        self.cumsums.zero_()
        self.buffer.zero_()
        self.Ns.zero_()
        self.cnt = 0

    @property
    def N(self):
        return self.Ns.sum()

    def __str__(self) -> str:
        return "{}([{:8d}]  u={:6.2f}, avgN={:6.2f})".format(
            self.name, self.cnt, self.stats()[0], self.Ns.mean()
        )


def get_parametrized_policy(env, opt):
    """ Configures a parametrized policy. """

    obs_size = env.observation_space.shape[-1]
    try:
        act_size = env.action_space.shape[-1]
    except IndexError:
        act_size = env.action_space.n

    if isinstance(env.action_space, gym.spaces.Box):
        policy = TanhNormal(act_size)
        estimator = ActorCritic(obs_size, act_size * 2, policy, **opt.estimator.args)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        policy = Discrete(act_size)
        estimator = ActorCritic(obs_size, act_size, policy, **opt.estimator.args)
    else:
        raise NotImplementedError("Unknown action space.")

    return estimator


def compute_nstep_returns(next_value, rewards, masks, *args, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def compute_gae_returns(next_value, rewards, masks, *args, gamma=0.99, lmbd=0.99):
    values = args[0] + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lmbd * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def validate(opt, env_fn, policy):
    env = env_fn.x()
    policy.eval()
    ep_R = []

    for _ in range(opt.valid_ep_cnt):
        obs, done, R = env.reset(), False, 0
        while not done:
            if opt.valid_deterministic_policy:
                with torch.no_grad():
                    logprobs = policy.actor(obs)
                    actions, _ = torch.split(logprobs, logprobs.shape[-1] // 2, -1)
            else:
                with torch.no_grad():
                    pi, _, _ = policy(obs)
                    actions = pi.sample()
            obs, reward, done, _ = env.step(actions)
            R += reward
        ep_R.append(R)

    ep_R = torch.tensor(ep_R)
    return Result(avgR=ep_R.mean(), N=len(ep_R), full=ep_R)


def pvalidate(env, policy, opt):
    """ Validation routine """
    env.eval()
    policy.eval()
    monitor = ParallelAverager(len(env), opt.device, name="R/ep")

    obs = env.reset()
    while True:
        with torch.no_grad():
            pi, _, _ = policy(obs)
        obs, reward, done, _ = env.step(pi.sample())
        monitor.put(reward, done)
        if monitor.N >= opt.valid_ep_cnt:
            break
    return monitor.stats()


def sampler(B, trajectories, max_batch_num):
    permutation = torch.randperm(trajectories[0].size(0))
    batch_idxs = torch.split(permutation, B)  # list of batches of transition idxs
    batch_idxs = batch_idxs[: (max_batch_num or len(batch_idxs))]
    for idxs in batch_idxs:
        yield [el[idxs, :] for el in trajectories]


def a2c_update(opt, agent, trajectories):
    model, optimizer = agent

    for _ in range(opt.mini_epochs):
        for batch in sampler(opt.batch_size, trajectories, opt.max_batch_num):

            obs, action, _, returns, advantage = batch

            pi, values, entropies = model(obs)

            action.clamp_(-0.99, 0.99)
            log_probs = pi.log_prob(action)

            # fmt: off
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = F.smooth_l1_loss(values, returns)
            entropy = entropies.mean()
            loss = actor_loss + critic_loss - opt.entropy_coeff * entropy
            # fmt: on

            optimizer.zero_grad()
            loss.backward()
            if opt.clip_pi_grad_norm:
                nn.utils.clip_grad_norm_(
                    model.actor.parameters(), opt.clip_pi_grad_norm
                )
            optimizer.step()


def ppo_update(opt, agent, trajectories):
    model, optimizer = agent

    # compute the advantage from the gae returns and the values
    obs, actions, log_probs, returns, old_values = trajectories
    advantages = returns - old_values
    if opt.normalize_advantage:
        advantages = (advantages - advantages.mean()) / advantages.std()
    trajectories = [obs, actions, log_probs, returns, advantages]

    # do the updates
    for _ in range(opt.mini_epochs):
        for batch in sampler(opt.batch_size, trajectories, opt.max_batch_num):

            obs, action, old_log_probs, returns, advantage = batch

            pi, values, entropies = model(obs)

            action.clamp_(-0.99, 0.99)
            new_log_probs = pi.log_prob(action)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - opt.clip, 1.0 + opt.clip) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropies.mean()
            critic_loss = (returns - values).pow(2).mean()
            loss = critic_loss + actor_loss + opt.entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            if opt.clip_pi_grad_norm:
                nn.utils.clip_grad_norm_(
                    model.actor.parameters(), opt.clip_pi_grad_norm
                )
            optimizer.step()


AGENTS = {
    "PPO": (compute_gae_returns, ppo_update),
    "PG": (compute_nstep_returns, a2c_update),
}


class Buffer:
    def __init__(self, max_size) -> None:
        self.max_size = max_size
        self.size = 0
        self.mem = []

    def put(self, observation, action, reward, log_prob, value, done):
        assert observation.ndim == 2, "Buffer assumes parallel environments."
        self.mem.append(
            [
                observation,
                action,
                reward.unsqueeze(1),
                log_prob,
                value,
                (1 - done).unsqueeze(1),
            ]
        )
        self.size += observation.shape[0]

    def flush(self):
        batch = [list(x) for x in zip(*self.mem[:-1])]
        del self.mem[:]  # flush the memory
        self.size = 0  # restart counter
        return batch

    def __len__(self):
        return self.size


def learn(env, model, mem, optimizer, opt, obs):
    env.train()
    model.train()
    monitor = ParallelAverager(len(env), opt.device, name="R/ep")

    compute_returns, update = AGENTS[opt.agent.name]

    train_step_cnt = 0
    while train_step_cnt < opt.train_step_cnt:

        # sample the env
        with torch.no_grad():
            pi, value, _ = model(obs)
            action = pi.sample()

        obs_, reward, done, _ = env.step(action)
        log_prob = pi.log_prob(action)

        # append stuff
        reward *= opt.agent.reward_scale or 1.0  # reward scale
        mem.put(obs, action, reward, log_prob, value, done)

        # logging, bookeeping
        train_step_cnt += obs.shape[0]
        monitor.put(reward, done)

        if len(mem) == opt.agent.dataset_size:
            observations, actions, rewards, log_probs, values, masks = mem.flush()

            with torch.no_grad():
                _, next_value, _ = model(obs_)
                returns = compute_returns(
                    next_value,
                    rewards,
                    masks,
                    values,
                    gamma=opt.agent.gamma,
                    lmbd=opt.agent.lmbd,
                )

            # bring everything to size
            returns = torch.cat(returns)
            observations, actions, rewards, log_probs, values, masks = [
                torch.cat(x)
                for x in (observations, actions, rewards, log_probs, values, masks)
            ]

            update(
                opt.agent,
                (model, optimizer),
                (observations, actions, log_probs, returns, values),
            )

        # next
        obs = obs_

    return monitor.stats(), obs


def trace_and_log(epoch, steps, trn_stats, val_stats):
    # Logging sucks
    rlog.trace(
        step=steps,
        val_R_ep=val_stats.avgR,
        trn_R_ep=trn_stats.avgR,
        val_eps=val_stats.N,
        trn_eps=trn_stats.N,
        epoch=epoch,
    )
    rlog.info(
        "{:8d}  valid: R/ep={:6.2f} std={:6.2f}  |  train: R/ep={:6.2f} std={:6.2f}".format(
            steps,
            val_stats.avgR,
            val_stats.full.std(),
            trn_stats.avgR,
            trn_stats.full.std(),
        )
    )


class PytorchContext(SpawnContext):
    def SimpleQueue(self):
        return mp.SimpleQueue()

    def Queue(self, maxsize=0):
        return mp.Queue(maxsize, ctx=self.get_context())


def run(opt):
    # torch.autograd.set_detect_anomaly(True)

    opt.device = torch.device(opt.env.args.get("device"))
    ioutil.create_paths(opt)
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)

    env = get_env(**opt.env.args)
    model = get_parametrized_policy(env, opt).to(opt.device)
    optimizer = getattr(O, opt.optim.name)(model.parameters(), **opt.optim.args)
    mem = Buffer(opt.agent.dataset_size)

    # validation stuff
    def make_env():
        def _thunk():
            return get_env(name=opt.env.args.get("name"), device=opt.device, env_no=1)

        return CloudpickleWrapper(_thunk)

    val_fn = partial(validate, opt, make_env())
    val_exec = ProcessPoolExecutor(max_workers=1, mp_context=PytorchContext())

    rlog.info(ioutil.config_to_string(opt))
    rlog.info(
        "\n{}: {}\n{}: {:4d}\n{}: {:3d}\n{}: {:3d}\n{}: {}".format(
            clr("ENV", "green"),
            env,
            clr("Dset size", "green"),
            opt.agent.dataset_size,
            clr("Trajectory length", "green"),
            opt.agent.dataset_size // len(env),
            clr("Batches available/epoch", "green"),
            opt.agent.dataset_size // opt.agent.batch_size,
            clr("Batches used/epoch", "green"),
            opt.agent.max_batch_num or "all available",
        )
    )
    rlog.info(model)

    val_job = None
    obs = env.reset()
    for epoch in range(opt.epochs):
        trn_stats, obs = learn(env, model, mem, optimizer, opt, obs)

        if val_job is None:
            val_job = val_exec.submit(val_fn, deepcopy(model))
            rlog.info("First is on the house.")
            continue

        rlog.info(f"Mai ești în viață? A: {val_job.running()}")
        rlog.info("Ask for results")
        val_stats = val_job.result()
        rlog.info("Results delivered")
        val_job = val_exec.submit(val_fn, deepcopy(model))

        # log and save
        steps = opt.train_step_cnt * epoch
        trace_and_log(epoch, steps, prev_trn_stats, val_stats)
        ioutil.checkpoint_agent(
            opt.out_dir, steps, estimator=model, cfg=opt, verbose=False,
        )
        prev_trn_stats = trn_stats

    # last train
    steps = opt.train_step_cnt * (epoch + 1)
    val_stats = val_job.result()
    trace_and_log(epoch + 1, steps, prev_trn_stats, val_stats)
    ioutil.checkpoint_agent(
        opt.out_dir, steps, estimator=model, cfg=opt, verbose=False,
    )


def main():
    """ Liftoff """
    run(parse_opts())
    # opt = ioutil.read_config("./configs/msl_ppo_llc.yaml")
    # run(opt)


if __name__ == "__main__":
    main()
