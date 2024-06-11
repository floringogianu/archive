import torch


class A2CLearnerMetaWorldNormalDist:
    def __init__(self, rollout_buffer, model, config):
        self.rollout_buffer = rollout_buffer
        self.model = model

        self.action_size = config.task.action_size
        self.beta_entropy = config.rl_algorithm.beta_entropy[0]
        self.beta_entropy_annealing = config.rl_algorithm.beta_entropy[1]
        if self.beta_entropy_annealing:
            self.beta_entropy_annealing_value = (
                self.beta_entropy - config.rl_algorithm.beta_entropy_final_value
            ) / config.training.nr_episodes
        self.beta_value_function_loss = config.rl_algorithm.beta_value_function_loss
        self.gamma = config.rl_algorithm.gamma

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), **config.optimizer.args
        )
        self.gradient_clipping = config.rl_algorithm.gradient_clipping[0]
        if self.gradient_clipping:
            self.gradient_clipping_value = config.rl_algorithm.gradient_clipping[1]

        self.optimization_step = 0
        self.epoch_length = config.training.epoch_length
        self.actor_losses = [None] * self.epoch_length
        self.critic_losses = [None] * self.epoch_length
        self.entropy_regularizations = [None] * self.epoch_length

        self.policy_dist_type = config.rl_algorithm.policy_dist_type
        self.l1_loss = torch.nn.L1Loss(reduction="none")

        self.gae = config.rl_algorithm.gae
        if self.gae:
            # includes both the discount and lambda for gae
            self.gamma_lambda = config.rl_algorithm.gae_lambda * self.gamma

    def step(self, R):
        self.optimizer.zero_grad()

        samples = self.rollout_buffer.get_data()

        # shape: rollout_length x batch x 1
        V = torch.stack(samples[0])
        r = torch.stack(samples[1])

        # shape: rollout_length x batch x action_size
        a_idx = torch.stack(samples[2])
        a_idx.clamp_(-0.99, 0.99)
        pi = samples[3]
        entropy = torch.stack(samples[4])
        entropy = entropy.mean(dim=2).mean(dim=1)

        # getting enropy and log probability from the policy
        log_p = []
        for i in range(len(pi)):
            log_p.append(pi[i].log_prob(a_idx[i]))
        log_p = torch.stack(log_p)
        # log_p and entropy -> shape = rollout_length
        # mean along the action_dimensions and afterwards along the batch dimension
        log_p = log_p.mean(dim=2).mean(dim=1)

        # computing advantages
        advantage = []
        returns = []
        for i in reversed(range(V.shape[0])):
            R = r[i] + self.gamma * R
            if self.gae and len(advantage) != 0:
                delta = r[i] + self.gamma * V[i + 1] - V[i]
                advantage.append(delta + self.gamma_lambda * advantage[-1])
            else:
                advantage.append(R - V[i])
            returns.append(R)
        # reverse the advantage list to be in the corect sequential order
        advantage.reverse()
        # advantage -> shape = rollout_length
        # mean along the batch dimension
        advantage = torch.stack(advantage).squeeze().mean(dim=1)
        returns.reverse()
        # returns and V should have the same shape => used for the critic loss
        returns = torch.stack(returns).squeeze()
        V = V.squeeze()

        actor_loss = self.loss_function_actor(log_p=log_p, advantage=advantage)
        critic_loss = self.loss_function_critic(returns=returns, values=V)
        entropy_regularization = self.entropy_regularization_function(entropy=entropy)
        loss = actor_loss + critic_loss + entropy_regularization

        # print(f'loss: {loss}, actor_loss: {actor_loss}, critic_loss:{critic_loss}, entropy_regularization:{entropy_regularization}')
        loss.backward(retain_graph=False)

        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_clipping_value
            )
        self.optimizer.step()

        self.actor_losses[
            self.optimization_step % self.epoch_length
        ] = actor_loss.item()
        self.critic_losses[
            self.optimization_step % self.epoch_length
        ] = critic_loss.item()
        self.entropy_regularizations[
            self.optimization_step % self.epoch_length
        ] = entropy_regularization.item()

        self.optimization_step += 1
        if self.beta_entropy_annealing:
            self.beta_entropy -= self.beta_entropy_annealing_value

    def loss_function_actor(self, log_p, advantage):
        return -(log_p * advantage.detach()).sum()

    def loss_function_critic(self, returns, values):
        return (
            self.beta_value_function_loss
            * self.l1_loss(returns, values).mean(dim=1).sum()
        )

    def entropy_regularization_function(self, entropy):
        return -self.beta_entropy * entropy.sum()
