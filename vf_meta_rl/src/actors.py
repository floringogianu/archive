import numpy as np
import torch

import src.policies as policies
from src.task_generators import VectorizedMetaWorldML1EnvironmentsSingleTask


class ContinuousActionsAgentNormalDistSingleTask:
    def __init__(self, rollout_buffer, model, config):
        # Set the model and the buffer used to save the observed transitions
        self.model = model
        self.rollout_buffer = rollout_buffer

        self.observation_size = config.task.observation_size
        self.action_size = config.task.action_size
        self.batch_size = config.meta_learning.meta_batch_size

        self.task_generator = VectorizedMetaWorldML1EnvironmentsSingleTask(
            config=config
        )
        self.get_new_task()

        # Load training parameters
        self.episode_length = config.meta_learning.rollout_max_length

        self.policy = policies.TanhNormal(self.action_size)

    def step(self):
        pi, V = self.model(
            observation=self.o_t,
            action=self.a_t,
            reward=self.r_t,
            t=self.t / self.episode_length,
        )

        pi_dis, entropy = self.policy(pi)
        # Vector of binary samples representing max/min value for each of the
        # action dimensions

        self.a_t = pi_dis.sample()
        env_a_t = self.a_t.numpy()
        # Step the environoment with the sampled action
        self.o_t, self.r_t, self.done, _, _ = self.env.step(env_a_t)

        # Convert the numpy values returned by the MetaWorld environment to
        # torch tensors used for the input of the LSTM
        # simplest normalization for metaworld (r in [0,10])
        self.r_t = torch.from_numpy(self.r_t).unsqueeze(dim=1) / self.max_reward_value
        self.o_t = torch.from_numpy(self.o_t)

        # there is no truncation in metaworld upon reaching the desired target,
        # so the only problem that could appear are very large observations
        if torch.any(self.o_t > 1e4):
            print(f"Last observation: {self.o_t}")
            raise Exception("Observation values are too big!")

        self.rollout_buffer.push(
            v=V, r=self.r_t, a_idx=self.a_t, pi=pi_dis, entropy=entropy
        )
        self.t += 1
        # MetaWorld envs don't signal that the episode should be truncated ....
        if self.t == self.episode_length:
            self.done = np.ones(self.batch_size, dtype=np.bool)

    def run_steps(self, nr_steps):
        for _ in range(nr_steps):
            self.step()
        if self.t == self.episode_length:
            return torch.zeros_like(self.r_t)
        else:
            V = self.model.get_next_state_value(
                observation=self.o_t,
                action=self.a_t,
                reward=self.r_t,
                t=self.t / self.episode_length,
            )
            return V

    def run_steps_until_done(self):
        while not np.any(self.done):
            self.step()
        return torch.zeros_like(self.r_t)

    def get_new_task(self):
        self.env = self.task_generator.change_tasks()
        self.reset_environments()
        self.model.reset_hidden_state()  # Reset LSTM hidden state

    def reset_environments(self):
        self.o_t, _ = self.env.reset()  # Reset environment
        self.o_t = torch.from_numpy(self.o_t)
        self.a_t = torch.zeros((self.batch_size, self.action_size), dtype=torch.long)
        self.r_t = torch.zeros((self.batch_size, 1))
        self.t = 0
        self.done = np.zeros(self.batch_size, dtype=np.bool_)

    def detach_model_hidden_state(self):
        self.model.detach_hidden_state()

    def run_episodes(
        self, tasks_set, n_exploration_eps=10, n_test_eps=1, return_trajectories=False
    ):
        tasks = self.task_generator.change_tasks()
        nr_tasks = tasks.action_space.shape[0]

        if return_trajectories:
            observations = []
            rewards = []
            infos = []
            dones = []
            actions = []
        with torch.no_grad():
            self.model.reset_hidden_state_eval_batch(batch_size=nr_tasks)
            for eps in range(n_exploration_eps + n_test_eps):
                o_t, _ = tasks.reset()  # Reset environment
                o_t = torch.from_numpy(o_t)
                a_t = torch.zeros((nr_tasks, self.action_size), dtype=torch.long)
                r_t = torch.zeros((nr_tasks, 1))
                t = 0
                done = np.zeros(nr_tasks, dtype=np.bool_)
                return_ = 0
                for t in range(self.episode_length):
                    pi, _ = self.model(
                        observation=o_t,
                        action=a_t,
                        reward=r_t,
                        t=t / self.episode_length,
                    )
                    pi_dis, _ = self.policy(pi)
                    a_t = pi_dis.sample()

                    env_a_t = a_t.numpy()
                    o_t, r_t, done, _, info = tasks.step(
                        env_a_t
                    )  # Step the environoment with the sampled action
                    if eps >= n_exploration_eps and return_trajectories:
                        observations.append(o_t)
                        rewards.append(r_t)
                        infos.append(info)
                        dones.append(done)
                        actions.append(a_t)
                    # Convert the numpy values returned by the MetaWorld environment to torch tensors used for the input of the LSTM
                    # simplest normalization for metaworld (r in [0,10])
                    r_t = torch.from_numpy(r_t).unsqueeze(dim=1) / self.max_reward_value
                    o_t = torch.from_numpy(o_t)
                    if eps >= n_exploration_eps:
                        return_ += r_t
        if return_trajectories:
            return (
                return_.mean().item(),
                info["success"].mean().item() * 100,
                observations,
                rewards,
                actions,
                infos,
                dones,
            )
        else:
            return return_.mean().item(), info["success"].mean().item() * 100
