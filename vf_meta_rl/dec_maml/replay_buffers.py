import torch
from numpy import random
from configparser import ConfigParser

    # batch = batch_size x hist_len x observation_size (for states -> s=batch[0], s_=batch[3])
    #         obs_element x batch_size x action_size (for action -> a=batch[1])
    #         obs_element x batch_size x 1 (for the other elements -> r=batch[2], terminal=batch[4])

class ExperienceReplay_episodic:

    def __init__(
        self,
        config: ConfigParser
    ) -> None:
        
        self.memory = []
        self.capacity = config.getint('MODEL','replay_capacity')
        self.batch_size = config.getint('MODEL','batch_size')
        self.device = torch.device(config.get('MODEL','replay_device'))

        self.warmup_steps = config.getint('MODEL','warmup_steps')
        self.multistep_R = config.getboolean('MODEL','multistep_R')
        if self.multistep_R:
            self.n_step = config.getint('MODEL','n_step')
        self.prioritized_ER = config.getboolean('MODEL','prioritized_ER')
        if self.prioritized_ER:
            self.max_priority = config.getfloat('MODEL','max_priority')

        self.position = 0
        self._size = 0
        
    def push(self, transition: list) -> None:
        if self.prioritized_ER:
            transition.append(self.max_priority)
        if self._size < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position += 1
        self._size = max(self._size, self.position)
        self.position = self.position%self.capacity

    def sample(self) -> list:
        # 1-step TD
        if self.prioritized_ER:
            print('Not implemented!')
        else:
            transitions = [], [], [], []
            samples_idx = random.randint(0,self._size,(self.batch_size,)).tolist()
            for idx in samples_idx:
                transition = self.memory[idx]
                transitions[0].append(transition[0])
                transitions[1].append(transition[1])
                transitions[2].append(transition[2])
                transitions[3].append(transition[3])
        return transitions

    @property
    def is_ready(self) -> bool:
        return self._size >= self.warmup_steps


class ExperienceReplay_tensorBuffer_episodic:

    def __init__(
        self,
        action_size: int,
        observation_size: int,
        config: ConfigParser
    ) -> None:
        self.capacity = config.getint('MODEL','replay_capacity')
        self.batch_size = config.getint('MODEL','batch_size')
        self.device = torch.device(config.get('MODEL','replay_device'))

        self.observation_buffer = torch.empty(size=(self.capacity,observation_size),dtype=torch.float,device=self.device)
        self.next_observation_buffer = torch.empty(size=(self.capacity,observation_size),dtype=torch.float,device=self.device)
        self.action_buffer = torch.empty(size=(self.capacity,action_size),dtype=torch.float,device=self.device)
        self.reward_buffer = torch.empty(size=(self.capacity,1),dtype=torch.float,device=self.device)
        self.terminal_buffer = torch.empty(size=(self.capacity,1),dtype=torch.bool,device=self.device)

        self.warmup_steps = config.getint('MODEL','warmup_steps')
        self.multistep_R = config.getboolean('MODEL','multistep_R')
        if self.multistep_R:
            self.n_step = config.getint('MODEL','n_step')
        self.prioritized_ER = config.getboolean('MODEL','prioritized_ER')
        if self.prioritized_ER:
            self.max_priority = config.getfloat('MODEL','max_priority')

        self.position = 0
        self._size = 0
        
    def push(self, transition: list) -> None:
        if self.prioritized_ER:
            transition.append(self.max_priority)
        self.observation_buffer[self.position] = torch.from_numpy(transition[0])
        self.action_buffer[self.position] = transition[1]
        self.reward_buffer[self.position] = transition[2]
        self.next_observation_buffer[self.position] = torch.from_numpy(transition[3])
        self.terminal_buffer[self.position] = transition[4] 

        self.position += 1
        self._size = max(self._size, self.position)
        self.position = self.position%self.capacity

    def sample(self) -> list:
        # 1-step TD
        if self.prioritized_ER:
            print('Not implemented!')
        else:
            samples_idx = random.randint(0,self._size,(self.batch_size,)).tolist()
            print(type(samples_idx))
            transitions = (self.observation_buffer[samples_idx],self.action_buffer[samples_idx],self.reward_buffer[samples_idx],
                        self.next_observation_buffer[samples_idx],self.terminal_buffer[samples_idx]) ############## !!!!!!!!!!!!! For loop
        return transitions

    @property
    def is_ready(self) -> bool:
        return self._size >= self.warmup_steps


        

        


        

        