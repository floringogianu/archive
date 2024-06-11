""" Entry point. """
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from numpy.lib.arraysetops import isin

import rlog
import torch
from torch import optim as O
from torch.nn.utils import spectral_norm

import src.io_utils as ioutil
from src.c51 import C51PolicyEvaluation, C51PolicyImprovement
from src.estimators import AtariNet
from src.replay import ExperienceReplay
from src.rl_routines import Episode, validate
from src.wrappers import get_wrapped_atari

# Default hyperparams
# Follows hyperparams from Dopamine:
# https://github.com/google/dopamine/blob/master/dopamine/agents/rainbow/configs/c51.gin
DEFAULT = ioutil.YamlNamespace(
    device=torch.device("cuda"),
    mem_device=torch.device("cpu"),
    epoch_cnt=200,
    train_step_cnt=250_000,
    valid_step_cnt=125_000,
    warmup_steps=20_000,
    update_freq=4,
    target_update_freq=8_000,
    batch_size=32,
    gamma=0.99,
    epsilon=(1.0, 0.01, 250_000, 20_000),
    val_epsilon=0.001,  # validation epsilone greedy
    support=(-10, 10, 51),
    channel_cnt=1,
    hist_len=4,
    lr=0.00025,
    optim_eps=0.0003125,
)


def train_one_epoch(
    env,
    agent,
    epoch_step_cnt,
    update_freq,
    target_update_freq,
    warmup_steps,
    total_steps=0,
    last_state=None,
):
    """ Policy iteration for a given number of steps. """

    replay, policy, policy_evaluation = agent
    while True:
        # do policy improvement steps for the length of an episode
        # if _state is not None then the environment resumes from where
        # this function returned.
        for transition in Episode(env, policy, _state=last_state):

            _state, _pi, reward, state, done = transition
            total_steps += 1

            # push one transition to experience replay
            replay.push((_state, _pi.action, reward, state, done))

            # learn if a minimum no of transitions have been pushed in Replay
            if total_steps >= warmup_steps:
                if total_steps % update_freq == 0:
                    # sample from replay and do a policy evaluation step
                    batch = replay.sample()
                    loss = policy_evaluation(batch).detach().item()
                    rlog.put(loss=loss, lrn_steps=batch[0].shape[0])

                if total_steps % target_update_freq == 0:
                    policy_evaluation.update_target_estimator()

            # some more stats
            rlog.put(trn_reward=reward, trn_done=done, trn_steps=1)
            if total_steps % 50_000 == 0:
                msg = "[{0:6d}] R/ep={trn_R_ep:2.2f}, tps={trn_tps:2.2f}"
                rlog.info(msg.format(total_steps, **rlog.summarize()))

            # exit if done
            if total_steps % epoch_step_cnt == 0:
                return total_steps, _state

        # This is important! It tells Episode(...) not to attempt to resume
        # an episode intrerupted when this function exited last time.
        last_state = None


def get_experiment_name(opt):
    args = {"snorm": opt.spectral, "gclip": opt.clip_grad_norm}
    name = "c51"
    for k, v in args.items():
        if v:
            if isinstance(v, bool):
                name += f"_{k}"
            else:
                name += f"_{k}:{v}"
    name += f"_{opt.game}"
    name += "_DEBUG" if opt.debug else ""
    return name


def main(opt):
    """ Entry point of the program. """

    ckpt = None
    if opt.resume:
        ckpt_path = opt.resume
        ckpt = ioutil.load_checkpoint(ckpt_path)
        opt = ioutil.dict_to_namespace(ckpt["cfg"])
        opt.resume = ckpt_path
    else:
        # configure experiment name based on program args
        opt.experiment = get_experiment_name(opt)

    assert (
        opt.warmup_steps == opt.epsilon[-1]
    ), "warmup steps and warmup epsilon should be equal."

    ioutil.create_paths(opt)

    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_R_ep", metargs=["trn_reward", "trn_done"]),
        rlog.SumMetric("trn_ep_cnt", metargs=["trn_done"]),
        rlog.AvgMetric("trn_loss", metargs=["trn_loss", 1]),
        rlog.FPSMetric("trn_tps", metargs=["trn_steps"]),
        rlog.FPSMetric("lrn_tps", metargs=["lrn_steps"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )

    # Initialize the objects we will use during training.

    env = get_wrapped_atari(opt.game, mode="train", device=opt.device)
    opt.action_cnt = env.action_space.n

    replay = ExperienceReplay(
        capacity=int(1e6),
        batch_size=opt.batch_size,
        hist_len=opt.hist_len,
        device=opt.mem_device,
    )
    estimator = AtariNet(
        opt.channel_cnt,
        opt.hist_len,
        opt.action_cnt,
        hidden_size=512,
        support=opt.support,  # this configures a C51 DQN estimator.
        spectral=opt.spectral,  # spectral normalization
    )
    print(estimator)
    target_estimator = deepcopy(estimator)
    estimator.to(opt.device)
    target_estimator.to(opt.device)
    optimizer = O.Adam(estimator.parameters(), lr=opt.lr, eps=opt.optim_eps)
    policy_evaluation = C51PolicyEvaluation(
        estimator,
        optimizer,
        opt.gamma,
        target_estimator=target_estimator,
        clip_grad_norm=opt.__dict__.get("clip_grad_norm", True),
    )
    policy_improvement = C51PolicyImprovement(
        estimator, opt.epsilon, opt.action_cnt
    )

    # if we loaded a checkpoint
    if ckpt is not None:
        # load state dicts
        estimator.load_state_dict(ckpt["estimator_state"])
        target_estimator.load_state_dict(ckpt["estimator_state"])
        optimizer.load_state_dict(ckpt["optim_state"])

        # sometimes the experiment is intrerupted while saving the replay
        # buffer and it gets corrupted. Therefore we attempt restoring
        # from the previous checkpoint and replay.
        try:
            replay.load(Path(opt.out_dir) / "replay.gz")
        except (SystemError, EOFError):
            rlog.info("Replay gzip at {} is faulty.".format(ckpt["step"]))
            prev_ckpt_idx = ckpt["step"] - opt.train_step_cnt
            rlog.info("Loading a previous snapshot @{}.".format(prev_ckpt_idx))
            ckpt = ioutil.load_checkpoint(opt.resume, idx=prev_ckpt_idx)
            replay.load(Path(opt.out_dir) / "prev_replay.gz")

        last_epsilon = None
        for _ in range(ckpt["step"]):
            last_epsilon = next(policy_improvement.epsilon)
        rlog.info(f"Last epsilon: {last_epsilon}.")
        # some counters
        last_epoch = ckpt["step"] // opt.train_step_cnt
        rlog.info(f"Resuming from epoch epoch {last_epoch}.")
        start_epoch = last_epoch + 1
        steps = ckpt["step"]
    else:
        steps = 0
        start_epoch = 1
        # add some hardware and git info, log and save
        opt = ioutil.add_platform_info(opt)

    rlog.info("\n" + ioutil.config_to_string(opt))
    ioutil.save_config(opt, opt.out_dir)

    # Start training

    last_state = None  # used by train_one_epoch to know how to resume episode.
    for epoch in range(start_epoch, opt.epoch_cnt + 1):

        # train for 250,000 steps
        steps, last_state = train_one_epoch(
            env,
            (replay, policy_improvement, policy_evaluation),
            opt.train_step_cnt,
            opt.update_freq,
            opt.target_update_freq,
            opt.warmup_steps,
            total_steps=steps,
            last_state=last_state,
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # validate for 125,000 steps
        validate(
            C51PolicyImprovement(
                policy_improvement.estimator, opt.val_epsilon, opt.action_cnt,
            ),
            get_wrapped_atari(opt.game, mode="testing", device=opt.device),
            opt.valid_step_cnt,
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # save the checkpoint
        ioutil.checkpoint_agent(
            opt.out_dir,
            steps,
            estimator=estimator,
            optim=optimizer,
            cfg=opt,
            replay=replay,
        )


if __name__ == "__main__":
    PARSER = ArgumentParser("Contrastive Value Functions.")
    PARSER.add_argument("game", type=str, help="Name of the game.")
    PARSER.add_argument(
        "-r",
        "--resume",
        dest="resume",
        default="",
        type=str,
        help="Path to the experiment to be resumed.",
    )
    PARSER.add_argument(
        "-s",
        "--spectral",
        dest="spectral",
        choices=[None, "conv", "middle", "full"],
        default=None,
        help="Choice of layers spectral normalization is used on.",
    )
    PARSER.add_argument(
        "--clip-grad-norm",
        dest="clip_grad_norm",
        action="store_true",
        help="Whether to clip the norm of the gradient.",
    )
    PARSER.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="If in debug mode.",
    )
    ARGS = PARSER.parse_args()
    # update the defaults with the values from the command line
    DEFAULT.__dict__.update(**ARGS.__dict__)
    main(DEFAULT)
