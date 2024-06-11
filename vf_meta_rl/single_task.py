import metaworld
import src.models as models
from liftoff import parse_opts
from src.a2c_learners import A2CLearnerMetaWorldNormalDist
from src.actors import ContinuousActionsAgentNormalDistSingleTask
from src.replay_buffers import RolloutBufferNormalDist
from src.trainers import TrainerMetaWorldSingleTask
from src.utils import set_all_seeds


def run(opt):
    set_all_seeds(opt.run_id)

    # initialize data regarding the selected task
    ml1 = metaworld.ML1(opt.task.name)
    env = ml1.train_classes[opt.task.name]()
    opt.task.action_size = env.action_space.shape[0]
    opt.task.action_high = env.action_space.high
    opt.task.action_low = env.action_space.low
    opt.task.episode_length = env.max_path_length
    opt.task.observation_size = env.observation_space.shape[0]
    del ml1, env

    if opt.model.init == "orthogonal":
        model = models.A2C_LSTM_Gaussian_OrthogonalInit(
            observation_size=opt.task.observation_size,
            action_size=opt.task.action_size,
            hidden_size=opt.model.hidden_size,
            batch_size=opt.meta_learning.meta_batch_size,
        )
    elif opt.model.init == "default":
        model = models.A2C_LSTM_Gaussian(
            observation_size=opt.task.observation_size,
            action_size=opt.task.action_size,
            hidden_size=opt.model.hidden_size,
            batch_size=opt.meta_learning.meta_batch_size,
        )
    rollout_buffer = RolloutBufferNormalDist()
    learner = A2CLearnerMetaWorldNormalDist(
        rollout_buffer=rollout_buffer, model=model, config=opt
    )
    actor = ContinuousActionsAgentNormalDistSingleTask(
        rollout_buffer=rollout_buffer, model=model, config=opt
    )
    trainer = TrainerMetaWorldSingleTask(actor=actor, learner=learner, config=opt)

    trainer.train_agent()


def main():
    run(parse_opts())


if __name__ == "__main__":
    main()
