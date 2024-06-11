from gymnasium import Env
import numpy as np


class environment_state_observations:
    def __init__(self,environment:Env):
        self.environment = environment
        self.current_state = None

    def reset(self) -> None:
        observation, _ = self.environment.reset()
        self.current_state = observation

    def step(self, action:np.ndarray, action_idx:int) -> list:
        observation, reward, terminated, truncated, _ = self.environment.step(action)
        transition = [self.current_state,action_idx,reward,observation,terminated or truncated]
        self.current_state = observation
        return transition 