import learners
import actors
import trainers
import environments
import replay_buffers
import os
import gymnasium as gym
import configparser

def main():
    config = configparser.ConfigParser()
    config.read('/root/RL/factorised_meta_learning/config.ini')

    gym_environment = gym.make(config.get('ENVIRONMENT','task_name'))
    val_gym_environment = gym.make(config.get('ENVIRONMENT','task_name'))
    
    environment = environments.environment_state_observations(environment=gym_environment)
    replay = replay_buffers.ExperienceReplay_episodic(config=config)
    learner = learners.DQN_state_observations(replay=replay,environment=gym_environment,config=config)
    actor = actors.basic_actor_discrete_actions(environment=environment,replay=replay,learner=learner,config=config)

    trainer = trainers.train_state_observations(actor=actor,learner=learner,validation_environment=val_gym_environment,config=config)
    trainer.train_agent()


if __name__ == "__main__":
    main()