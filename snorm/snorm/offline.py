""" Entry point. """
from liftoff import parse_opts

from termcolor import colored as clr
import rlog
import src.io_utils as ioutil
from src.agents import AGENTS
from src.wrappers import get_env

from src.training import experiment_factory, train_offline_one_epoch, validate


def run(opt):
    """ Entry point of the program. """

    if __debug__:
        print(
            clr(
                "Code might have assertions. Use -O in liftoff when running stuff.",
                color="red",
                attrs=["bold"],
            )
        )

    assert (
        opt.game in opt.replay_path
    ), "Data was collected on a different game than the one you are evaluating on."

    ioutil.create_paths(opt)

    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_loss", metargs=["trn_loss", 1]),
        rlog.FPSMetric("lrn_tps", metargs=["lrn_steps"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )

    # Initialize the objects we will use during training.
    env, (replay, policy_improvement, policy_evaluation) = experiment_factory(
        opt, offline=True
    )

    guts = [
        env,
        replay,
        policy_evaluation.estimator,
        policy_evaluation.optimizer,
        policy_improvement,
        policy_evaluation,
    ]
    rlog.info(("\n\n{}" * len(guts)).format(*guts))

    if opt.estimator.args.get("spectral", None) is not None:
        for k in policy_evaluation.estimator.get_spectral_norms().keys():
            # k = f"min{str(k)[1:]}"
            rlog.addMetrics(rlog.ValueMetric(k, metargs=[k]))

    # add some hardware and git info, log and save
    opt = ioutil.add_platform_info(opt)

    rlog.info("\n" + ioutil.config_to_string(opt))
    ioutil.save_config(opt, opt.out_dir)

    # Start training

    steps = 0
    start_epoch = 1
    for epoch in range(start_epoch, opt.epoch_cnt + 1):

        replay.reload_N(N=opt.replay_cnt, workers=int(opt.omp_num_threads or 1))
        rlog.info(f"Sampled N={opt.replay_cnt} replay buffers.")

        # train for 250,000 steps (well, actually 250k / 4)
        steps = train_offline_one_epoch(
            (replay, policy_improvement, policy_evaluation),
            opt.train_step_cnt,
            opt.agent.args["update_freq"],
            opt.agent.args["target_update_freq"],
            rlog.getRootLogger(),
            total_steps=steps,
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # validate for 125,000 steps
        validate(
            AGENTS[opt.agent.name]["policy_improvement"](
                policy_improvement.estimator, opt.action_cnt, epsilon=opt.val_epsilon
            ),
            get_env(opt, mode="testing"),
            opt.valid_step_cnt,
            rlog.getRootLogger(),
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # save the checkpoint
        if opt.save:
            ioutil.checkpoint_agent(
                opt.out_dir,
                steps,
                estimator=policy_evaluation.estimator,
                target_estimator=policy_evaluation.target_estimator,
                optim=policy_evaluation.optimizer,
                cfg=opt,
                replay=None,
                save_every_replay=False,
            )


def main():
    """ Liftoff
    """
    run(parse_opts())


if __name__ == "__main__":
    main()
