from argparse import ArgumentParser
from pathlib import Path

import torch
from numpy import random

import rlog
import src.io_utils as ioutil
from src.estimators import AtariNet
from src.wrappers import get_wrapped_atari
from src.rl_routines import Episode, validate
from src.utils import EpsilonGreedyOutput
from src.c51 import C51PolicyImprovement


class RandomPolicy:
    def __init__(self, act_no=18):
        self.act_no = act_no

    def act(self, x):
        return EpsilonGreedyOutput(
            action=random.randint(0, self.act_no), q_value=0, full=0
        )


def random_validation(opt):
    """ Validation routine """
    rlog.info("Starting validation...")
    env = get_wrapped_atari(
        opt.game,
        mode="testing",
        seed=opt.seed,
        no_gym=opt.no_gym,
        device=opt.mem_device,
    )
    policy = RandomPolicy(act_no=env.action_space.n)

    done_eval, step_cnt = False, 0
    with torch.no_grad():
        while not done_eval:
            for _, _, reward, _, done in Episode(env, policy):
                rlog.put(reward=reward, done=done, val_frames=1)

                step_cnt += 1
                if step_cnt >= opt.valid_step_cnt:
                    done_eval = True
                    break

    env.close()


def main(args):
    root = Path(args.path)
    opt = ioutil.YamlNamespace(
        experiment="cvf",
        seed=42,
        game="seaquest",
        device=torch.device("cuda"),
        valid_step_cnt=125_000,
        train_step_cnt=250_000,
        support=(-10, 10, 51),
        channel_cnt=1,
        hist_len=4,
        val_epsilon=0.001,
        out_dir=root / "sticky_area",
    )
    try:
        opt.out_dir.mkdir()
    except FileExistsError:
        print("Folder already exists...")

    print(opt)

    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True)
    rlog.addMetrics(
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )

    env = get_wrapped_atari(opt.game, mode="testing", device=opt.device)
    opt.action_cnt = env.action_space.n
    estimator = AtariNet(
        opt.channel_cnt,
        opt.hist_len,
        opt.action_cnt,
        hidden_size=512,
        support=opt.support,  # this configures a C51 DQN estimator.
    )
    estimator.to(opt.device)

    # Get paths to checkpoints
    ckpt_paths = sorted(root.glob("*model*"))

    for epoch, p in enumerate(ckpt_paths):

        # load checkpoint
        estimator.load_state_dict(torch.load(p))
        rlog.info("Loading ckpt {} from {}".format(epoch, str(p)))

        # do a validation run
        # TODO: parallel!
        validate(
            C51PolicyImprovement(estimator, opt.val_epsilon, opt.action_cnt),
            env,
            opt.valid_step_cnt,
        )

        # random_validation(opt)
        rlog.traceAndLog((epoch + 1) * opt.train_step_cnt)


if __name__ == "__main__":
    PARSER = ArgumentParser(description="Read checkpoints and do validation.")
    PARSER.add_argument("path", type=str, help="Path to experiment.")
    main(PARSER.parse_args())
