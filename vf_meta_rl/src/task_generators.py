import random

import gymnasium as gym
import metaworld


class VectorizedMetaWorldML1EnvironmentsSingleTask:
    def __init__(self, config):
        self.task_name = config.task.name
        self.nr_tasks = config.meta_learning.meta_batch_size

        # Construct the benchmark, sampling tasks
        self.ml1 = metaworld.ML1(self.task_name)
        self.envs = []
        for _ in range(self.nr_tasks):
            # Create an environment with selected task
            env = self.ml1.train_classes[self.task_name]()
            self.envs.append(env)
        self.task = random.choice(self.ml1.train_tasks)

    def generate_task(self, task, env_idx: int):
        self.envs[env_idx].set_task(task)
        return self.envs[env_idx]

    def change_tasks(self):
        env_fns = []
        for i in range(self.nr_tasks):
            env_fns.append(lambda: self.generate_task(task=self.task, env_idx=i))
        tasks = gym.vector.AsyncVectorEnv(env_fns=env_fns)
        return tasks


def main():
    from argparse import Namespace

    cfg = Namespace(
        task=Namespace(name="reach-v2"), meta_learning=Namespace(meta_batch_size=8)
    )
    tasks = VectorizedMetaWorldML1EnvironmentsSingleTask(cfg)
    env = tasks.change_tasks()
    print(env.reset())


if __name__ == "__main__":
    main()
