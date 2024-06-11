from os.path import join
import torch
from torch.utils.tensorboard import SummaryWriter


class TrainerMetaWorldSingleTask:
    def __init__(self, actor, learner, config):
        self.actor = actor
        self.learner = learner

        self.epoch_length = config.training.epoch_length
        self.nr_training_epochs = config.training.nr_training_epochs

        self.data_dir = config.out_dir
        tensorboard_dir = config.paths.tensorboard_logdir
        self.writer = SummaryWriter(tensorboard_dir)
        self.run_id = config.title + "; seed " + str(config.run_id)

        self.model_checkpoint_frequency = config.training.model_checkpoint_frequency
        self.episode_length = config.meta_learning.rollout_max_length
        self.rollouts_per_task = config.meta_learning.rollouts_per_task
        self.rollout_max_length = config.meta_learning.rollout_max_length
        self.n_steps = config.rl_algorithm.n_steps

    def train_agent(self) -> list:
        epoch = 0
        episode = 0
        step = 0
        returns_train = []
        success_rates_train = []
        self.actor.get_new_task()
        while epoch < self.nr_training_epochs:
            R = self.actor.run_steps(nr_steps=self.n_steps)
            step += self.n_steps
            self.learner.step(R=R)

            if step % self.rollout_max_length == 0:
                episode += 1
                if episode % self.epoch_length == 0:
                    epoch += 1
                    print(f"Epoch {epoch} done.")

                    actor_loss = sum(self.learner.actor_losses) / len(
                        self.learner.actor_losses
                    )
                    self.writer.add_scalar(
                        "%s/actor_loss" % (self.run_id), actor_loss, epoch
                    )
                    critic_loss = sum(self.learner.critic_losses) / len(
                        self.learner.critic_losses
                    )
                    self.writer.add_scalar(
                        "%s/critic_loss" % (self.run_id), critic_loss, epoch
                    )
                    entropy_regularization = sum(
                        self.learner.entropy_regularizations
                    ) / len(self.learner.entropy_regularizations)
                    self.writer.add_scalar(
                        "%s/entropy_regularization" % (self.run_id),
                        entropy_regularization,
                        epoch,
                    )

                    return_, success_rate = self.actor.run_episodes("train")
                    returns_train.append(return_)
                    success_rates_train.append(success_rate)
                    self.writer.add_scalar(
                        "%s/train_return" % (self.run_id), returns_train[-1], epoch
                    )
                    self.writer.add_scalar(
                        "%s/train_success_rate" % (self.run_id),
                        success_rates_train[-1],
                        epoch,
                    )

                    if epoch % 10 == 0:
                        torch.save(
                            {
                                "returns_train": returns_train,
                                "success_rates_train": success_rates_train,
                            },
                            self.data_dir + "/loss_and_return.pt",
                        )
                    if epoch % self.model_checkpoint_frequency == 0:
                        torch.save(
                            (
                                self.learner.model.state_dict(),
                                self.learner.optimizer.state_dict(),
                            ),
                            join(self.data_dir, "model_epoch_" + str(epoch) + ".pt"),
                        )

                if episode % self.rollouts_per_task == 0:
                    self.actor.get_new_task()
                else:
                    self.actor.reset_environments()
                    self.actor.detach_model_hidden_state()
            else:
                self.actor.detach_model_hidden_state()

        # Save the final results
        torch.save(
            {
                "returns_train": returns_train,
                "success_rates_train": success_rates_train,
            },
            self.data_dir + "/loss_and_return.pt",
        )
        torch.save(
            (self.learner.model.state_dict(), self.learner.optimizer.state_dict()),
            join(self.data_dir, "model_epoch_" + str(epoch) + ".pt"),
        )

        return returns_train
