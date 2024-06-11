import torch
import models
from collections import OrderedDict
from configparser import ConfigParser
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN_state_observations:
    def __init__(self,replay,environment,config:ConfigParser):
        self.replay = replay
        observation_size = environment.observation_space.shape[0]
        action_size = config.getint('ENVIRONMENT','nr_discrete_actions')**environment.action_space.shape[0]
        self.model = models.basic_mlp(observation_size,action_size,config.getint('MODEL','hidden_size'))#.to(device)
        self.target_model = models.basic_mlp(observation_size,action_size,config.getint('MODEL','hidden_size'))#.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.target_model_update_frequency = config.getint('MODEL','target_model_update_frequency')
        self.gamma = config.getfloat('MODEL','gamma')
        self.loss_function = torch.nn.MSELoss()
       
        lr = config.getfloat('OPTIMIZER','lr')
        betas = [config.getfloat('OPTIMIZER','beta1'),config.getfloat('OPTIMIZER','beta2')]
        epsilon = config.getfloat('OPTIMIZER','epsilon')
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr,betas=betas,eps=epsilon)
        
        self.double_Q = config.getboolean('MODEL','double_Q')
        self.prioritized_ER = config.getboolean('MODEL','prioritized_ER')
        self.multistep_R = config.getboolean('MODEL','multistep_R')
        if self.multistep_R:
            self.n_step = config.getint('MODEL','n_step')
            self.n_step_gamma = (self.gamma**torch.arange(start=0,stop=self.n_step-1,step=1)).unsqueeze(0)

        self.optimization_step = 0
        self.epoch_length = config.getint('TRAINING','epoch_length') 
        self.losses = [None]*self.epoch_length 
   
        

    def step(self):
        self.optimizer.zero_grad()

        if self.multistep_R:
            print('Not implemented!')
        else:
            samples = self.replay.sample()
            s = torch.tensor(np.array(samples[:][0]), dtype=torch.float)
            a = torch.tensor(np.array(samples[:][1])).unsqueeze(0)
            r = torch.tensor(samples[:][2],dtype=torch.float)
            s_ = torch.tensor(np.array(samples[:][3]), dtype=torch.float)

            if self.double_Q:
                print('Not implemented!')
            else:
                q = self.model(s)
                q = torch.gather(input=q,dim=1,index=a).squeeze() 
                with torch.no_grad():
                    q_, _ = self.target_model(s_).max(dim=1)
                    q_ = r + self.gamma*q_
                loss = self.loss_function(q,q_)
                loss.backward()
                self.optimizer.step()
                self.losses[self.optimization_step%self.epoch_length] = loss.item()
            
            self.optimization_step += 1
            if self.optimization_step%self.target_model_update_frequency==0:
                self.update_target_model()
           
    def get_actor_parameters(self) -> OrderedDict:
        return self.model.state_dict()

    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())


################################################################################################################################################

# class DecQN_state_observations:
#     def __init__(self,replay,observation_size:int,action_size:int,log_dir:str,config:ConfigParser):
#         self.replay = replay
#         self.model = models.basic_mlp(observation_size,action_size,config.getint('MODEL','hidden_size'))
#         self.target_model = models.basic_mlp(observation_size,action_size,config.getint('MODEL','hidden_size'))
#         self.target_model.load_state_dict(self.model.state_dict())
#         for param in self.target_model.parameters():
#             param.requires_grad = False
#         self.target_model_update_frequency = config.getint('MODEL','target_model_update_frequency')
#         self.gamma = config.getfloat('MODEL','gamma')
#         self.lr = config.getfloat('OPTIMIZER','lr')
#         self.loss_function = torch.nn.MSELoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
#         self.double_Q = config.getboolean('MODEL','double_Q')
#         self.prioritized_ER = config.getboolean('MODEL','prioritized_ER')
#         self.multistep_R = config.getboolean('MODEL','multistep_R')
#         if self.multistep_R:
#             self.n_step = config.getint('MODEL','n_step')
#             self.n_step_gamma = (self.gamma**torch.arange(start=0,stop=self.n_step-1,step=1)).unsqueeze(0)

#         self.optimization_step = 0 
#         self.losses = [] 

#         self.nr_discrete_actions = config.getint('ENVIRONMENT','nr_discrete_actions')
#         self.batch_size = config.getint('MODEL','batch_size')
        

#     def step(self):
#         self.optimizer.zero_grad()

#         if self.multistep_R:
#             print('Not implemented!')
#         else:
#             samples = self.replay.sample()
#             s = torch.tensor(np.array(samples[:][0]), dtype=torch.float)
#             a = torch.tensor(np.array(samples[:][1]))
#             r = torch.tensor(samples[:][2])
#             s_ = torch.tensor(np.array(samples[:][3]), dtype=torch.float)

#             if self.double_Q:
#                 print('Not implemented!')
#             else:
#                 q = self.model(s).view(self.batch_size,-1,self.nr_discrete_actions)
#                 q = r + torch.gather(input=q,dim=2,index=a) 
#                 with torch.no_grad():
#                     q_ = self.target_model(s_).view(self.batch_size,-1,self.nr_discrete_actions).max(dim=2)
#                 loss = self.loss_function(q,q_)
#                 loss.backward()
#                 self.optimizer.step()

#                 self.losses.append(loss.item())
            
#             self.optimization_step += 1
#             if self.optimization_step%self.target_model_update_frequency==0:
#                 self.update_target_model()
            
#     def get_actor_parameters(self) -> OrderedDict:
#         return self.model.state_dict()

#     def update_target_model(self) -> None:
#         self.target_model.load_state_dict(self.model.state_dict())