"""Register the envs in OpenAI Gym.
"""
from gym.envs.registration import register

from .centipede import __all__ as env_definitions

for env_def in env_definitions:
    env_name = env_def[:-3]  # SomethingSomethingEnv  -->  SomethingSomething
    register(
        id=f"{env_name}-v2",
        entry_point=f"src.centipede_env.centipede:{env_def}",
        max_episode_steps=1000,
        reward_threshold=6000,
    )
    print(f"Registered {env_name}.")
