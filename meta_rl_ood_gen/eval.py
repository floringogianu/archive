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
        entropy_scheduler=None,
        clip_grad_norm=True,
    ) -> None:
        self.act_num = act_num
        self.pi = policy
        self.optim = optim
        self.n_steps = n_steps
        self.gamma = gamma
        self.entropy_scheduler = entropy_scheduler or repeat(0.05)
        self.clip_grad_norm = clip_grad_norm
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
        self.beta = next(self.entropy_scheduler)

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
        critic_loss = F.mse_loss(values, Gts, reduction="sum")
        loss = policy_loss + 0.5 * critic_loss - self.beta * entropies.sum()

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
        _reward = 0 if self.step_cnt == 0 else self.transitions[-1].reward
        _action = 0 if self.step_cnt == 0 else self.transitions[-1].action
        act_emb = self.act_emb[_action]
        return torch.cat([torch.tensor([observation, _reward]), act_emb])

    @classmethod
    def from_opt(cls, opt):
        obs_dim, act_num, hid_dim = 1, 2, 48
        obs_dim = obs_dim + act_num + 1  #  obs, act, reward
        policy = Policy(act_num, obs_dim, hid_dim=hid_dim)
        optim = getattr(torch.optim, opt.optim.name)(
            policy.parameters(), **opt.optim.args
        )
        opt.entropy_scheduler[-1] *= opt.total_steps
        entropy_scheduler = linear_schedule(*opt.entropy_scheduler)
        return cls(
            act_num,
            policy,
            optim,
            entropy_scheduler=entropy_scheduler,
            **opt.args,
            clip_grad_norm=opt.optim.clip_grad_norm,
        )


def validate(agent, tasks, split, valid_episodes=300, value_log=None):
    with torch.no_grad():
        for _ in range(valid_episodes):
            agent.reset()
            tid, bandit = next(tasks)
            (_observation, done) = bandit.reset()
            while not done:
                action = agent.step(_observation)
                observation, reward, done = bandit.step(action)
                agent.learn(
                    _observation, action, reward, observation, done, training=False
                )
                _observation = observation
                if value_log:
                    value_log.put(
                        regret=bandit.regret,
                        h_regret=bandit.hregret,
                        acc=bandit.acc,
                        reward=reward,
                    )
            # stats
            if split == "train":
                rlog.put(
                    trn_regret=bandit.regret,
                    trn_h_regret=bandit.hregret,
                    trn_acc=bandit.acc,
                )
            elif split == "valid":
                rlog.put(
                    val_regret=bandit.regret,
                    val_h_regret=bandit.hregret,
                    val_acc=bandit.acc,
                )
            elif split == "test":
                rlog.put(
                    tst_regret=bandit.regret,
                    tst_h_regret=bandit.hregret,
                    tst_acc=bandit.acc,
                )


def run(opt):
    opt.seed = opt.seed or torch.randint(1_000_000, (1,)).item()
    utl.seed_everything(opt.seed)

    rlog.init("eval", opt.out_dir, tensorboard=True, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_avgR", metargs=["trn_regret", 1]),
        rlog.AvgMetric("trn_avgHR", metargs=["trn_h_regret", 1]),
        rlog.AvgMetric("trn_avgAcc", metargs=["trn_acc", 1]),
        rlog.AvgMetric("val_avgR", metargs=["val_regret", 1]),
        rlog.AvgMetric("val_avgHR", metargs=["val_h_regret", 1]),
        rlog.AvgMetric("val_avgAcc", metargs=["val_acc", 1]),
        rlog.AvgMetric("tst_avgR", metargs=["tst_regret", 1]),
        rlog.AvgMetric("tst_avgHR", metargs=["tst_h_regret", 1]),
        rlog.AvgMetric("tst_avgAcc", metargs=["tst_acc", 1]),
    )
    xtra = rlog.getLogger("eval.test")
    xtra.addMetrics(
        rlog.ValueMetric("regret", metargs=["regret"]),
        rlog.ValueMetric("h_regret", metargs=["h_regret"]),
        rlog.ValueMetric("acc", metargs=["acc"]),
        rlog.ValueMetric("reward", metargs=["reward"]),
    )

    # print
    utl.maybe_sample_hyperparams_(opt)
    rlog.info(utl.config_to_string(opt, newl=True))

    trn_tasks = Tasks(opt.env.N, "train", spec=opt.env.spec)
    val_tasks = Tasks(opt.env.N, "valid", spec=opt.env.spec)
    tst_tasks = Tasks(opt.env.N, "test", spec=opt.env.spec)

    opt.agent.total_steps = trn_tasks.N * opt.trials * opt.episodes
    agent = A2C.from_opt(opt.agent)

    # load checkpoint
    ckpt = torch.load(f"{opt.out_dir}/final_model.th")
    agent.pi.load_state_dict(ckpt["policy_state"])

    # start evaluation
    rlog.info("Eval on train")
    validate(agent, trn_tasks, "train", valid_episodes=500)
    rlog.info("Eval on valid")
    validate(agent, val_tasks, "valid", valid_episodes=500)
    rlog.info("Eval on test")
    validate(agent, tst_tasks, "test", valid_episodes=500, value_log=xtra)
    rlog.traceAndLog(20_000)
    xtra.traceAndLog(500 * 100)  # number of episodes / number of trials


if __name__ == "__main__":
    run(parse_opts())
