from collections import defaultdict, namedtuple
from copy import deepcopy
from itertools import chain, repeat

import rlog
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from liftoff import parse_opts

import src.util as utl
from src.tasks import Tasks


class Policy(nn.Module):
    def __init__(self, act_num, obs_dim, hid_dim=48) -> None:
        """Uses a RNN to parametrize a policy.
        Warning: Assumes NO batch dimension.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.rnn = nn.LSTMCell(obs_dim, hid_dim)
        self.hx, self.cx = None, None
        self.policy_net = nn.Linear(hid_dim, act_num)
        self.value_net = nn.Linear(hid_dim, 1)

    def forward(self, observation):
        self.hx, self.cx = self.rnn(observation, (self.hx, self.cx))
        logits = self.policy_net(self.hx)
        value = self.value_net(self.hx)
        return D.Categorical(logits=logits), value

    def reset(self, soft=False):
        if soft:
            self.hx = self.hx.detach()
            self.cx = self.cx.detach()
        else:
            self.hx = torch.zeros(self.hid_dim)
            self.cx = torch.zeros(self.hid_dim)


def float_range(start, end, step):
    x = start
    if step > 0:
        while x < end:
            yield x
            x += step
    else:
        while x > end:
            yield x
            x += step


def linear_schedule(start, end, steps_no):
    step = (end - start) / (steps_no - 1.0)
    schedules = [float_range(start, end, step), repeat(end)]
    return chain(*schedules)


Transition = namedtuple("Transition", "t, pi, action, value, reward", defaults=[None])


class A2C:
    def __init__(
        self,
        act_num,
        policy,
        optim,
        n_steps=5,
        gamma=0.8,
        coeff_H=None,
        coeff_critic=0.5,
        clip_grad_norm=True,
        with_interaction_hist=True,
    ) -> None:
        self.act_num = act_num
        self.pi = policy
        self.optim = optim
        self.n_steps = n_steps
        self.gamma = gamma
        self.coeff_H = coeff_H or repeat(0.05)
        self.coeff_critic = coeff_critic
        self.clip_grad_norm = clip_grad_norm
        self.with_interaction_hist = with_interaction_hist
        self.act_emb = torch.eye(act_num)
        # agent state
        self.step_cnt = 0
        self.transitions = []
        self._temp_trans = None

    def reset(self, soft=False):
        if not soft:
            self.step_cnt = 0
            self.transitions.clear()
            self._temp_trans = None
        self.pi.reset(soft=False)

    def step(self, observation):
        # get policy and state value
        pi, value = self.pi(self._get_observation(observation))
        # update internal state of the agent
        action = pi.sample().item()
        self._temp_trans = Transition(self.step_cnt, pi, action, value.view(1))
        return action

    def learn(self, observation, action, reward, observation_, done, training=True):
        self.step_cnt += 1
        self.beta = next(self.coeff_H)

        # complete the transition, remove the temporary and add to buffer
        transition = Transition(*self._temp_trans[:-1], reward.item())
        self._temp_trans = None
        self.transitions.append(transition)

        # learn
        if self.step_cnt % self.n_steps == 0 or done:
            if training:
                self._update(done, observation_)

            # we keep the last transition
            # we need the reward and action at t-1 in the inputs
            del self.transitions[:-1]

    def _update(self, done, observation_):
        # compute logprobs, entropies and returns
        logprobs, entropies, values, rewards = [], [], [], []
        for _, pi, action, value, reward in self.transitions[-self.n_steps :]:
            logprobs.append(pi.log_prob(torch.tensor(action)).view(1))
            entropies.append(pi.entropy().view(1))
            values.append(value)
            rewards.append(reward)
        Gts = self._get_returns(done, observation_, rewards)

        # lists2tensors
        values = torch.cat(values)
        logprobs = torch.cat(logprobs)
        entropies = torch.cat(entropies)
        advantages = Gts - values

        # losses
        policy_loss = (-logprobs * advantages.detach()).sum()
        critic_loss = self.coeff_critic * F.mse_loss(values, Gts, reduction="sum")
        loss = policy_loss + critic_loss - self.beta * entropies.sum()

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.clip_grad_norm)
        self.optim.step()

    def _get_returns(self, done, obs_, rewards):
        R = 0
        if not done:
            with torch.no_grad():
                R = self.pi(self._get_observation(obs_))[1].item()
        Gts = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            Gts.insert(0, R)
        return torch.tensor(Gts)

    def _get_observation(self, observation):
        # check if initial state
        # and initialize some dummy values
        if not self.with_interaction_hist:
            return torch.tensor([observation], dtype=torch.float32)
        _reward = 0 if self.step_cnt == 0 else self.transitions[-1].reward
        _action = 0 if self.step_cnt == 0 else self.transitions[-1].action
        act_emb = self.act_emb[_action]
        return torch.cat([torch.tensor([observation, _reward]), act_emb])

    @classmethod
    def from_opt(cls, opt):
        obs_dim, act_num, hid_dim = 1, 2, 48
        # add the reward and the action embeddings sizes to the observation size
        wih = opt.args["with_interaction_hist"]
        obs_dim = obs_dim + act_num + 1 if wih else obs_dim
        policy = Policy(act_num, obs_dim, hid_dim=hid_dim)
        optim = getattr(torch.optim, opt.optim.name)(
            policy.parameters(), **opt.optim.args
        )

        start, end, ratio = opt.args["coeff_H"]
        agent_kws = {k:v for k,v in opt.args.items() if k != "coeff_H"}
        coeff_H = linear_schedule(*[start, end, ratio * opt.total_steps])
        return cls(
            act_num,
            policy,
            optim,
            coeff_H=coeff_H,
            clip_grad_norm=opt.optim.clip_grad_norm,
            **agent_kws,
        )

    def __str__(self):
        attrs = ", ".join(
            [
                f"{k}={v}"
                for k, v in self.__dict__.items()
                if isinstance(v, (int, float, bool, str))
            ]
        )
        return f"A2C({attrs})"


def validate(agent, tasks, total_eval_tasks=300):
    with torch.no_grad():
        for _ in range(total_eval_tasks):
            agent.reset()
            _, bandit = next(tasks)
            (_observation, done) = bandit.reset()
            while not done:
                action = agent.step(_observation)
                observation, reward, done = bandit.step(action)
                agent.learn(
                    _observation, action, reward, observation, done, training=False
                )
                _observation = observation

            # stats, depending on the split
            metrics = ["regret", "hregret", "acc"]
            rlog.put(**{f"{tasks.split}_{m}": getattr(bandit, m) for m in metrics})


def run(opt):
    opt.seed = opt.seed or torch.randint(1_000_000, (1,)).item()
    utl.seed_everything(opt.seed)

    rlog.init("MetaSG", opt.out_dir, tensorboard=True, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_regret", metargs=["train_regret", 1]),
        rlog.AvgMetric("trn_hRegret", metargs=["train_hregret", 1]),
        rlog.AvgMetric("trn_acc", metargs=["train_acc", 1]),
        rlog.AvgMetric("val_regret", metargs=["valid_regret", 1]),
        rlog.AvgMetric("val_hRegret", metargs=["valid_hregret", 1]),
        rlog.AvgMetric("val_acc", metargs=["valid_acc", 1]),
        rlog.AvgMetric("tst_regret", metargs=["test_regret", 1]),
        rlog.AvgMetric("tst_hRegret", metargs=["test_hregret", 1]),
        rlog.AvgMetric("tst_acc", metargs=["test_acc", 1]),
        # rlog.ValueMetric("trn_regret", metargs=["trn_regret"]),
        # rlog.ValueMetric("trn_h_regret", metargs=["trn_h_regret"]),
        # rlog.ValueMetric("trn_acc", metargs=["trn_acc"]),
    )

    # sample hyperparameters if it's the case and print
    utl.maybe_sample_hyperparams_(opt)

    # get the bandits
    trn_tasks = Tasks("train", opt.env.spec, **opt.env.args)
    val_tasks = Tasks("valid", opt.env.spec, **opt.env.args)
    tst_tasks = Tasks("test", opt.env.spec, **opt.env.args)

    total_trials = opt.env.args["total_trials"]
    opt.agent.total_steps = opt.total_tasks * opt.total_eps_per_task * total_trials
    trn_agent = A2C.from_opt(opt.agent)
    val_agent = A2C.from_opt(opt.agent)

    # save the resulting config
    rlog.info(utl.config_to_string(opt, newl=True))
    rlog.info(trn_agent)
    utl.save_config(opt, opt.out_dir)

    # start experiment
    for tsk_cnt in range(opt.total_tasks):
        trn_agent.reset()
        _, bandit = next(trn_tasks)
        for _ in range(opt.total_eps_per_task):
            trn_agent.reset(soft=True)
            (_observation, done) = bandit.reset()
            while not done:
                action = trn_agent.step(_observation)
                observation, reward, done = bandit.step(action)
                trn_agent.learn(_observation, action, reward, observation, done)
                _observation = observation

        if tsk_cnt % 1000 == 0 and tsk_cnt != 0:
            # use another agent, just to make sure the state
            # of the one we are training is not changed
            val_agent.pi.load_state_dict(trn_agent.pi.state_dict())
            # validate on all splits
            validate(val_agent, trn_tasks, total_eval_tasks=opt.total_eval_tasks)
            validate(val_agent, val_tasks, total_eval_tasks=opt.total_eval_tasks)
            validate(val_agent, tst_tasks, total_eval_tasks=opt.total_eval_tasks)
            rlog.traceAndLog(step=tsk_cnt)

            # save final weights only
            torch.save(
                {"policy_state": trn_agent.pi.state_dict()},
                f"{opt.out_dir}/model_{tsk_cnt:08d}.th",
            )


if __name__ == "__main__":
    run(parse_opts())
