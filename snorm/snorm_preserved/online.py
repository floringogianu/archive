""" Entry point. """
import gc
import sys
from functools import reduce
from pathlib import Path

import rlog
import torch
from liftoff import parse_opts
from torch import optim as O

import src.estimators as E
import src.io_utils as ioutil
from src.agents import AGENTS
from src.estimators.layers.utils import streamline_model
from src.replay import ExperienceReplay
from src.rl_routines import Episode
from src.wrappers import get_env


def train_one_epoch(
    env,
    agent,
    epoch_step_cnt,
    update_freq,
    target_update_freq,
    total_steps=0,
    last_state=None,
):
    """ Policy iteration for a given number of steps. """

    replay, policy, policy_evaluation = agent
    online_net = policy_evaluation.estimator
    target_net = policy_evaluation.target_estimator
    online_net.train()
    target_net.train()
    streamline_model(online_net, True)
    streamline_model(target_net, True)

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
            if replay.is_ready:
                if total_steps % update_freq == 0:
                    # sample from replay and do a policy evaluation step
                    batch = replay.sample()

                    streamline_model(online_net, False)
                    streamline_model(target_net, False)

                    loss = policy_evaluation(batch).detach().item()
                    rlog.put(trn_loss=loss, lrn_steps=batch[0].shape[0])

                    streamline_model(online_net, True)
                    streamline_model(target_net, True)

                if total_steps % target_update_freq == 0:
                    policy_evaluation.update_target_estimator()

            # some more stats
            rlog.put(trn_reward=reward, trn_done=done, trn_steps=1)
            if (policy.estimator.spectral is not None) and (total_steps % 1000 == 0):
                rlog.put(**policy.estimator.get_spectral_norms())
            if total_steps % 50_000 == 0:
                msg = "[{0:6d}] R/ep={trn_R_ep:2.2f}, tps={trn_tps:2.2f}"
                rlog.info(msg.format(total_steps, **rlog.summarize()))

            # exit if done
            if total_steps % epoch_step_cnt == 0:
                return total_steps, _state

        # This is important! It tells Episode(...) not to attempt to resume
        # an episode intrerupted when this function exited last time.
        last_state = None


def validate(policy, env, steps):
    """ Validation routine """
    policy.estimator.eval()
    streamline_model(policy.estimator, True)

    done_eval, step_cnt = False, 0
    with torch.no_grad():
        while not done_eval:
            for _, _, reward, _, done in Episode(env, policy):
                rlog.put(reward=reward, done=done, val_frames=1)
                step_cnt += 1
                if step_cnt >= steps:
                    done_eval = True
                    break
    env.close()
    streamline_model(policy.estimator, False)


def experiment_factory(opt):
    """ Configures an environment and an agent """
    env = get_env(opt)
    opt.action_cnt = action_cnt = env.action_space.n

    # we use the warmup steps in the epsilon-greedy schedule to set the warmup
    # in the replay. If epsilon-greedy is not a schedule the warmup will be
    # given either replay.args.warmup_steps or by batch-size
    if isinstance(opt.agent.args["epsilon"], list):
        opt.replay["warmup_steps"] = opt.agent.args["epsilon"][-1]
    replay = ExperienceReplay(**opt.replay)

    assert (
        replay.warmup_steps == opt.agent.args["epsilon"][-1]
    ), "warmup steps and warmup epsilon should be equal."

    if (opt.agent.name == "DQN") and ("support" in opt.estimator.args):
        raise ValueError("DQN estimator should not have a support.")
    if (opt.agent.name == "C51") and ("support" not in opt.estimator.args):
        raise ValueError("C51 requires an estimator with support.")

    estimator_args = opt.estimator.args
    if opt.estimator.name == "MLP":
        estimator_args["layers"] = [
            reduce(lambda x, y: x * y, env.observation_space.shape),
            *estimator_args["layers"],
        ]
    if "MinAtar" in opt.game:
        estimator_args["input_ch"] = env.observation_space.shape[-1]
    estimator = getattr(E, opt.estimator.name)(action_cnt, **estimator_args)
    estimator.to(opt.device)

    optimizer = getattr(O, opt.optim.name)(estimator.parameters(), **opt.optim.args)
    policy_evaluation = AGENTS[opt.agent.name]["policy_evaluation"](
        estimator, optimizer, **opt.agent.args
    )
    policy_improvement = AGENTS[opt.agent.name]["policy_improvement"](
        estimator, action_cnt, **opt.agent.args
    )
    return env, (replay, policy_improvement, policy_evaluation)


def run(opt):
    """ Entry point of the program. """

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
    env, (replay, policy_improvement, policy_evaluation) = experiment_factory(opt)

    rlog.info("\n\n{}\n\n{}\n\n{}".format(env, replay, policy_evaluation.estimator))
    rlog.info("\n\n{}\n\n{}".format(policy_improvement, policy_evaluation))

    if opt.estimator.args.get("spectral", None) is not None:
        for k in policy_evaluation.estimator.get_spectral_norms().keys():
            # k = f"min{str(k)[1:]}"
            rlog.addMetrics(rlog.ValueMetric(k, metargs=[k]))

    # if we loaded a checkpoint
    if Path(opt.out_dir).joinpath("replay.gz").is_file():

        # sometimes the experiment is intrerupted while saving the replay
        # buffer and it gets corrupted. Therefore we attempt restoring
        # from the previous checkpoint and replay.
        try:
            idx = replay.load(Path(opt.out_dir) / "replay.gz")
            ckpt = ioutil.load_checkpoint(opt.out_dir, idx=idx)
            rlog.info(f"Loaded most recent replay (step {idx}).")
        except:
            gc.collect()
            rlog.info("Last replay gzip is faulty.")
            idx = replay.load(Path(opt.out_dir) / "prev_replay.gz")
            ckpt = ioutil.load_checkpoint(opt.out_dir, idx=idx)
            rlog.info(f"Loading a previous snapshot (step {idx}).")

        # load state dicts

        # load state dicts
        ioutil.special_conv_uv_buffer_fix(
            policy_evaluation.estimator, ckpt["estimator_state"]
        )
        policy_evaluation.estimator.load_state_dict(ckpt["estimator_state"])
        ioutil.special_conv_uv_buffer_fix(
            policy_evaluation.target_estimator, ckpt["target_estimator_state"]
        )
        policy_evaluation.target_estimator.load_state_dict(
            ckpt["target_estimator_state"]
        )
        policy_evaluation.optimizer.load_state_dict(ckpt["optim_state"])

        last_epsilon = None
        for _ in range(ckpt["step"]):
            last_epsilon = next(policy_improvement.epsilon)
        rlog.info(f"Last epsilon: {last_epsilon}.")
        # some counters
        last_epoch = ckpt["step"] // opt.train_step_cnt
        rlog.info(f"Resuming from epoch {last_epoch}.")
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
            total_steps=steps,
            last_state=last_state,
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # validate for 125,000 steps
        validate(
            AGENTS[opt.agent.name]["policy_improvement"](
                policy_improvement.estimator, opt.action_cnt, epsilon=opt.val_epsilon
            ),
            get_env(opt, mode="testing"),
            opt.valid_step_cnt,
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # save the checkpoint
        if opt.agent.save:
            ioutil.checkpoint_agent(
                opt.out_dir,
                steps,
                estimator=policy_evaluation.estimator,
                target_estimator=policy_evaluation.target_estimator,
                optim=policy_evaluation.optimizer,
                cfg=opt,
                replay=replay,
                save_replay=(epoch % 8 == 0 or epoch == opt.epoch_cnt),
            )


def main(args):
    """ Liftoff
    """

    if "--sesion-id" in sys.argv:
        opt = parse_opts()
    else:
        opt = ioutil.read_config(args[0])
    run(opt)


if __name__ == "__main__":
    main(sys.argv[1:])
