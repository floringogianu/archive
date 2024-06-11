import torch
from torch import nn


# Model used for continuous control tasks using Bernoulli distribution for the policy -> bang-bang control
class A2C_LSTM_Gaussian(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int,
        batch_size: int,
        obs_encoding="none",
        obs_encoding_hidden_size=-1,
    ):
        super(A2C_LSTM_Gaussian, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.obs_encoding = obs_encoding
        if self.obs_encoding == "none":
            self.obs_encoder = nn.Identity()
        elif self.obs_encoding == "mlp":
            self.obs_encoder = nn.Sequential(
                nn.Linear(
                    in_features=observation_size, out_features=obs_encoding_hidden_size
                ),
                nn.Tanh(),
                nn.Linear(
                    in_features=obs_encoding_hidden_size,
                    out_features=obs_encoding_hidden_size,
                ),
                nn.Tanh(),
            )
            observation_size = obs_encoding_hidden_size
        print(self.obs_encoder)
        print(observation_size)

        # LSTM used for meta-learning
        self.lstm = nn.LSTMCell(
            input_size=observation_size + action_size + 2, hidden_size=hidden_size
        )
        self.reset_hidden_state()
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=self.hidden_size, out_features=2 * action_size),
        )
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=self.hidden_size, out_features=1),
        )

    def forward(self, observation, action, reward, t):
        observation = self.obs_encoder(observation)
        lstm_input = torch.cat(
            (observation, action, reward, t * torch.ones(reward.shape)), dim=1
        ).float()
        self.hx, self.cx = self.lstm(lstm_input, (self.hx, self.cx))
        pi = self.actor(self.hx)
        V = self.critic(self.hx)
        return pi, V

    def get_next_state_value(self, observation, action, reward, t):
        with torch.no_grad():
            observation = self.obs_encoder(observation)
            lstm_input = torch.cat(
                (observation, action, reward, t * torch.ones(reward.shape)), dim=1
            ).float()
            hx, _ = self.lstm(lstm_input, (self.hx, self.cx))
            V = self.critic(hx)
        return V

    def reset_hidden_state(self):
        self.hx = torch.zeros(self.batch_size, self.hidden_size)
        self.cx = torch.zeros(self.batch_size, self.hidden_size)

    def detach_hidden_state(self):
        self.hx = self.hx.detach()
        self.cx = self.cx.detach()

    def reset_hidden_state_eval_batch(self, batch_size):
        self.hx = torch.zeros(batch_size, self.hidden_size)
        self.cx = torch.zeros(batch_size, self.hidden_size)
