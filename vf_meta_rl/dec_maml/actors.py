from configparser import ConfigParser
from itertools import product

import numpy as np
import torch
from gymnasium import Env


class basic_actor_discrete_actions:
    def __init__(self, replay, learner, environment: Env, config: ConfigParser):
        self.environment = environment
        self.replay = replay
        self.learner = learner
        self.policy = self.learner.model
        action_space = self.environment.environment.action_space
        nr_discrete_actions = config.getint("ENVIRONMENT", "nr_discrete_actions")
        actions = [
            [
                a_i
                for a_i in np.linspace(
                    start=action_space.low[i],
                    stop=action_space.high[i],
                    num=nr_discrete_actions,
                )
            ]
            for i in range(action_space.shape[0])
        ]
        self.actions = [action for action in product(*actions)]
        print(self.actions)
        self.epsilon = config.getfloat("MODEL", "epsilon_greedy")
        self.episode_length = config.getint("ENVIRONMENT", "episode_length")

    def step(self):
        with torch.no_grad():
            action_idx = self.epsilon_greedy()
        action = self.actions[action_idx]
        transition = self.environment.step(action=action, action_idx=action_idx)
        self.replay.push(transition)
        if transition[4]:
            self.environment.reset()

    def run_steps(self, nr_steps):
        for _ in range(nr_steps):
            self.step()

    def run_episode(self, environment: Env = None) -> float:
        # the environment parameter will be different from None only in the case of running val epsiodes (since the gym environment will be different)
        return_ = 0
        environment.reset()
        if environment is None:
            for _ in range(self.episode_length):
                with torch.no_grad():
                    action_idx = self.epsilon_greedy()
                action = self.actions[action_idx]
                return_ += self.environment.step(action=action, action_idx=action_idx)[
                    2
                ]
        else:
            for _ in range(self.episode_length):
                with torch.no_grad():
                    action_idx = self.epsilon_greedy()
                action = self.actions[action_idx]
                _, reward, _, _, _ = environment.step(action)
                return_ += reward
        return return_

    def run_and_log_episode(self, initial_state: torch.tensor) -> list:
        transitions = [], [], []
        self.environment.reset(initial_state)
        transitions[0].append(self.environment.current_state)
        for _ in range(self.episode_length):
            with torch.no_grad():
                action_idx = self.epsilon_greedy()
            action = self.actions[action_idx]
            transition = self.environment.step(action=action, action_idx=action_idx)
            transitions[0].append(transition[3])
            transitions[1].append(transition[1])
            transitions[2].append(transitions[2])
        return transitions

    def update_parameters(self):
        self.policy.load_state_dict(self.learner.get_actor_parameters())

    def replay_warmup(self):
        self.environment.reset()
        while not self.replay.is_ready:
            self.step()

    def epsilon_greedy(self) -> int:
        e = np.random.rand(1)
        if e < self.epsilon:
            return np.random.randint(low=0, high=len(self.actions))
        return torch.argmax(
            self.policy(
                torch.from_numpy(self.environment.current_state).to(dtype=torch.float)
            )
        ).item()
