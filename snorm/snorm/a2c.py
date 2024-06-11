""" Hope for the best """
import gym
import rlog
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as O
from liftoff import parse_opts
from torch import nn

import src.io_utils as ioutil
from src.common import get_env
from src.estimators import hook_spectral_normalization

# from brax.io import html
# from IPython.display import HTML


class TanhNormal:
    def __init__(self, action_num, min_std=0.0005) -> None:
        self.action_num = action_num
        self.min_std = min_std

    def __call__(self, x):

        loc, scale = torch.split(x, self.action_num, -1)
        scale = F.softplus(scale) + self.min_std

        normal = D.Normal(loc, scale)
        pi = D.TransformedDistribution(
            normal, [D.transforms.TanhTransform(cache_size=1)]
        )
        return pi, normal.entropy()

    def __repr__(self) -> str:
        return "tanh(Normal(loc: {a}, scale: {a}))".format(a=("B", self.action_num))


class Discrete:
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
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, actor_outdim, policy, layer_dims=None, **kwargs):
        super(ActorCritic, self).__init__()

        self.policy = policy

        self.actor = _make_mlp([num_inputs, *layer_dims, actor_outdim])
        self.critic = _make_mlp([num_inputs, *layer_dims, 1])
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

    def __str__(self):
        return super().__str__() + "\nPolicy={}".format(self.policy)


class ParallelAverager:
    def __init__(self, B, device, name=None) -> None:
        self.name = name or None
        self.cnt = 0
        self.cumsums = torch.zeros((B,), device=device)
        self.buffer = torch.zeros((B,), device=device)
        self.Ns = torch.zeros((B,), device=device)

    def put(self, value, N):
        self.buffer += value
        if N.any():
            to_avg = N == 1.0  # create a mask of series that can be averaged
            self.cumsums[to_avg] += self.buffer[to_avg]
            self.buffer[to_avg] = 0  # reset the buffer
            self.Ns += N
        self.cnt += 1

    def get(self):
        idxs = self.Ns.nonzero()
        cumsums = self.cumsums[idxs]
        Ns = self.Ns[idxs]
        return (
            (cumsums / Ns).mean().item(),
            {"full": cumsums / Ns, "Ns": self.Ns},
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
            self.name, self.cnt, self.get()[0], self.Ns.mean()
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


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def validate(env, policy, opt):
    """ Validation routine """
    policy.eval()
    monitor = ParallelAverager(opt.env_no, opt.device, name="R/ep")

    obs = env.reset()
    while True:
        with torch.no_grad():
            pi, _, _ = policy(obs)
        obs, reward, done, _ = env.step(pi.sample())
        monitor.put(reward, done)
        if monitor.N >= opt.valid_ep_cnt:
            break
    return monitor


def learn(env, model, optimizer, opt):
    model.train()
    B, nsteps = opt.env_no, opt.agent.nsteps
    monitor = ParallelAverager(B, opt.device, name="R/ep")

    obs, trn_cnt = env.reset(), 0
    while trn_cnt < opt.train_step_cnt:

        log_probs = []
        values = []
        rewards = []
        masks = []
        entropies = []

        for _ in range(nsteps):

            # sample the env
            pi, value, entropy = model(obs)
            action = pi.sample()
            obs_, reward, done, _ = env.step(action)

            log_prob = pi.log_prob(action)

            # append stuff
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward.unsqueeze(1))
            masks.append((1 - done).unsqueeze(1))
            entropies.append(entropy)

            obs = obs_

            monitor.put(reward, done)
            trn_cnt += 1
            if trn_cnt % 5000 == 0:
                rlog.info(monitor)

        _, next_value, _ = model(obs_)
        Gt = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(Gt).detach()
        values = torch.cat(values)
        entropy = torch.cat(entropies)

        log_probs = log_probs.view(B, nsteps, -1)
        returns = returns.view(B, nsteps, -1)
        values = values.view(B, nsteps, -1)
        entropy = entropy.view(B, nsteps, -1).mean()

        advantage = returns - values
        advantage = (advantage - advantage.mean()) / advantage.std()

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = F.smooth_l1_loss(values, returns, reduction="none").mean()

        loss = actor_loss + critic_loss - opt.agent.entropy_coeff * entropy

        optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
    return monitor


def run(opt):
    opt.device = torch.device(opt.device)
    ioutil.create_paths(opt)
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)

    B, T = opt.env_no, opt.train_step_cnt

    env = get_env(env_name=opt.game, env_no=B, device=opt.device)
    model = get_parametrized_policy(env, opt).to(opt.device)
    optimizer = getattr(O, opt.optim.name)(model.parameters(), **opt.optim.args)

    rlog.info(ioutil.config_to_string(opt))
    rlog.info(model)

    for epoch in range(1, opt.epochs + 1):
        trn_monitor = learn(env, model, optimizer, opt)
        val_monitor = validate(env, model, opt)

        # Logging sucks
        steps = epoch * T
        trn_avgR, trn_info = trn_monitor.get()
        val_avgR, val_info = val_monitor.get()
        rlog.trace(
            step=steps,
            val_R_ep=val_avgR,
            trn_R_ep=trn_avgR,
            val_eps=val_monitor.N,
            trn_eps=trn_monitor.N,
            frame=steps * B,
        )
        rlog.info(
            "{:8d}  valid: R/ep={:6.2f} std={:6.2f}  |  train: R/ep={:6.2f} std={:6.2f}".format(
                steps,
                val_avgR,
                val_info["full"].std(),
                trn_avgR,
                trn_info["full"].std(),
            )
        )
        # save model
        ioutil.checkpoint_agent(
            opt.out_dir, steps, estimator=model, cfg=opt,
        )


def main():
    """ Liftoff """
    run(parse_opts())


if __name__ == "__main__":
    main()
