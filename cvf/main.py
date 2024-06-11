""" Entry point. """
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import torch
from torch import optim as O

import rlog
import src.io_utils as ioutil
import src.transforms as T
from src.c51 import C51PolicyEvaluation, C51PolicyImprovement
from src.estimators import AtariNet
from src.replay import OfflineExperienceReplay
from src.rl_routines import validate
from src.schedules import get_schedule
from src.wrappers import get_wrapped_atari
from src.utils import to_device

# Default hyperparams
DEFAULT = ioutil.YamlNamespace(
    device=torch.device("cuda"),
    mem_device=torch.device("cpu"),
    epoch_cnt=200,
    train_step_cnt=250_000,
    valid_step_cnt=125_000,
    update_freq=4,
    target_update_freq=8000,  # agent steps following Dopamine convention
    batch_size=32,
    gamma=0.99,
    support=(-10, 10, 51),
    channel_cnt=1,
    hist_len=4,
    lr=0.00025,
    optim_eps=0.0003125,
    val_eps=0.001,
)


def train_one_epoch_contrastive(
    steps, replay, policy_evaluation, target_update_freq, contrastive
):
    encoder, augmentation, tau_schedule, beta = contrastive
    device = next(policy_evaluation.estimator.parameters()).device

    for step in range(1, steps + 1):
        batch = to_device(replay.sample(), device)

        # the C51 loss
        loss = policy_evaluation(batch, update=False, backward=False)

        # compute the contrastive loss
        with torch.no_grad():
            aug_state = augmentation(batch[0]).to(torch.device("cuda"))
            aug_qs_probs = encoder(aug_state, probs=True)
            log_qs_probs = loss.qs_probs

        c51_loss = loss.loss.mean()
        # aug_loss = kl(log_qs_probs, aug_qs_probs)
        aug_loss = -torch.sum(aug_qs_probs * log_qs_probs, 2).mean()

        # add the two losses and update the online network
        aug_loss = beta * aug_loss
        tot_loss = c51_loss + aug_loss
        tot_loss.backward()
        policy_evaluation.update_estimator()

        # update the encoder
        tau = next(tau_schedule)
        online_weights = policy_evaluation.estimator.parameters()
        encoder_weights = encoder.parameters()
        for op, ep in zip(online_weights, encoder_weights):
            ep.data = tau * ep + (1 - tau) * op.data

        # update the target network
        if step % target_update_freq == 0:
            policy_evaluation.update_target_estimator()

        # stats
        rlog.put(
            loss=tot_loss.detach().item(),
            aug_loss=aug_loss.detach().item(),
            c51_loss=c51_loss.detach().item(),
            trn_frames=batch[0].shape[0],
        )


def train_one_epoch(steps, replay, policy_evaluation, target_update_freq):
    for step in range(1, steps + 1):
        batch = replay.sample()

        loss = policy_evaluation(batch)
        rlog.put(loss=loss.detach().item(), trn_frames=batch[0].shape[0])

        if step % target_update_freq == 0:
            policy_evaluation.update_target_estimator()


def get_experiment_name_(opt):
    if opt.contrastive:
        opt.experiment = f"cvf_{opt.game}"
    else:
        opt.experiment = f"base_{opt.game}"


def main(opt):
    ckpt = None
    if opt.resume:
        ckpt_path = opt.resume
        ckpt = ioutil.load_checkpoint(ckpt_path)
        opt = ioutil.dict_to_namespace(ckpt["cfg"])
        opt.resume = ckpt_path
    else:
        # configure experiment name based on the state of the program
        get_experiment_name_(opt)

        if opt.debug:
            opt.experiment = f"{opt.experiment}_DEBUG"
            opt.buffer_cnt = 1  # so we not OOM on dev machines

    ioutil.create_paths(opt)

    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_loss", metargs=["loss", 1]),
        rlog.FPSMetric("trn_fps", metargs=["trn_frames"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )

    opt.root = Path(
        "./data/{}/{}/replay_logs".format(opt.game.capitalize(), opt.dset_id)
    )
    opt.action_cnt = get_wrapped_atari(opt.game).action_space.n

    # Initialize the objects we will use during training.

    replay = OfflineExperienceReplay(
        opt.root,
        batch_size=opt.batch_size,
        device=opt.mem_device,
        is_async=True,
    )
    estimator = AtariNet(
        opt.channel_cnt,
        opt.hist_len,
        opt.action_cnt,
        hidden_size=512,
        support=opt.support,  # this configures a C51 DQN estimator.
    )
    target_estimator = deepcopy(estimator)
    target_estimator.reset_parameters()
    estimator.to(opt.device)
    target_estimator.to(opt.device)
    optimizer = O.Adam(estimator.parameters(), lr=opt.lr, eps=opt.optim_eps)
    policy = C51PolicyEvaluation(
        estimator, optimizer, opt.gamma, target_estimator=target_estimator,
    )

    # compute the number of gradient steps per epoch
    gradient_steps = int(opt.train_step_cnt / opt.update_freq)
    target_update_freq = int(opt.target_update_freq / opt.update_freq)

    rlog.info(f"{gradient_steps} gradient steps / epoch")
    rlog.info(f"Update target net every {target_update_freq} gradient steps.")

    encoder = None
    if opt.contrastive:
        encoder = deepcopy(estimator)
        encoder.reset_parameters()
        transform = T.Compose(T.HorizontalShift(), T.Intensity())
        opt.tau_schedule = ioutil.dict_to_namespace(
            {
                "start": 0.996,
                "end": 1.0,
                "steps": opt.epoch_cnt * gradient_steps,
            }
        )
        tau_schedule = get_schedule(**vars(opt.tau_schedule))
        opt.contrastive_loss_weight = 0.5
        rlog.addMetrics(
            rlog.AvgMetric("trn_aug_loss", metargs=["aug_loss", 1]),
            rlog.AvgMetric("trn_c51_loss", metargs=["c51_loss", 1]),
        )

    # if we loaded a checkpoint
    if ckpt is not None:
        # load state dicts
        estimator.load_state_dict(ckpt["estimator_state"])
        target_estimator.load_state_dict(ckpt["estimator_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        # same for contrastive objects
        if opt.contrastive:
            encoder.load_state_dict(ckpt["encoder_state"])
            # consume the schedule up to the checkpointed step
            for _ in range(ckpt["step"]):
                last_tau = next(tau_schedule)
            rlog.info(f"Last tau: {last_tau}.")

        last_epoch = ckpt["step"] // opt.train_step_cnt
        rlog.info(f"Resuming from epoch epoch {last_epoch}.")
        start_epoch = last_epoch + 1
    else:
        start_epoch = 1

    rlog.info("\n" + ioutil.config_to_string(opt))

    # training loop  --------

    for epoch in range(start_epoch, opt.epoch_cnt + 1):

        steps = epoch * opt.train_step_cnt

        # reload N checkpoints in RAM. This amounts to N * 1e6 transitions
        replay.reload_N(N=opt.buffer_cnt)

        # train for 62500 steps
        if opt.contrastive:
            train_one_epoch_contrastive(
                gradient_steps,
                replay,
                policy,
                opt.target_update_freq,
                (
                    encoder,
                    transform,
                    tau_schedule,
                    opt.contrastive_loss_weight,
                ),
            )
        else:
            train_one_epoch(
                gradient_steps, replay, policy, opt.target_update_freq
            )
        rlog.traceAndLog(steps)

        # validate for 125000 steps
        validate(
            C51PolicyImprovement(
                deepcopy(policy.estimator), opt.val_eps, opt.action_cnt
            ),
            get_wrapped_atari(opt.game, mode="testing", device=opt.device),
            opt.valid_step_cnt,
        )

        rlog.traceAndLog(steps)

        # save the checkpoint
        ioutil.checkpoint_agent(
            opt.out_dir,
            steps,
            estimator=estimator,
            optim=optimizer,
            encoder=encoder,
            cfg=opt,
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
        "-c",
        "--contrastive",
        dest="contrastive",
        action="store_true",
        help="Contrastive training.",
    )
    PARSER.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="If in debug mode.",
    )
    PARSER.add_argument(
        "-e",
        "--replay-cnt",
        dest="buffer_cnt",
        default=5,
        type=int,
        help="No of Replay buffers loaded at once. Defaults to 5.",
    )
    PARSER.add_argument(
        "-i",
        "--replay-id",
        dest="dset_id",
        default=1,
        type=int,
        help="ID of the Dopamine Atari Dataset, from 1 to 5. Default=1",
    )
    ARGS = PARSER.parse_args()
    # update the defaults with the values from the command line
    DEFAULT.__dict__.update(**ARGS.__dict__)
    main(DEFAULT)
