from configparser import ConfigParser
from os import mkdir
from os.path import join

import torch
from gymnasium import Env
from torch.utils.tensorboard import SummaryWriter


class train_state_observations:
    def __init__(
        self, actor, learner, validation_environment: Env, config: ConfigParser
    ):
        self.validation_environment = validation_environment
        self.actor = actor
        self.learner = learner
        self.update_ratio = config.getint("MODEL", "update_ratio")
        self.epoch_length = config.getint("TRAINING", "epoch_length")
        self.nr_training_epochs = config.getint("TRAINING", "nr_training_epochs")
        self.nr_validation_episodes = config.getint(
            "TRAINING", "nr_validation_episodes"
        )
        self.data_dir = join(
            config.get("PATHS", "data_logdir"), config.get("RUN_ID", "run_id")
        )
        mkdir(self.data_dir)
        tensorboard_dir = config.get("PATHS", "tensorboard_logdir")
        self.writer = SummaryWriter(tensorboard_dir)
        self.run_id = config.get("RUN_ID", "run_id")
        self.model_checkpoint_frequency = config.getint(
            "TRAINING", "model_checkpoint_frequency"
        )
        self.val_environment_seed = config.getint("TRAINING", "val_environment_seed")

    def train_agent(self) -> list:
        epoch = 0
        returns = []
        losses = []
        self.actor.replay_warmup()
        while epoch < self.nr_training_epochs:
            self.actor.run_steps(self.update_ratio)
            self.learner.step()
            if self.learner.optimization_step % self.epoch_length == 0:
                epoch += 1
                losses.append(sum(self.learner.losses) / self.epoch_length)
                self.writer.add_scalar("%s/loss" % (self.run_id), losses[-1], epoch)
                returns.append(self.run_validation_episodes())
                self.writer.add_scalar(
                    "%s/val_return" % (self.run_id), returns[-1], epoch
                )
                if epoch % self.model_checkpoint_frequency == 0:
                    torch.save(
                        (
                            self.learner.model.state_dict(),
                            self.learner.optimizer.state_dict(),
                        ),
                        join(self.data_dir, "model_epoch_" + str(epoch) + ".pt"),
                    )

        torch.save(
            (self.learner.losses, returns), self.data_dir + "/loss_and_return.pt"
        )
        return returns

    def run_validation_episodes(self) -> float:
        mean_return = 0
        self.validation_environment.reset(seed=self.val_environment_seed)
        for _ in range(self.nr_validation_episodes):
            mean_return += self.actor.run_episode(
                environment=self.validation_environment
            )
        return mean_return / self.nr_validation_episodes
