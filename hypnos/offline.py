""" Entry point. """
import rlog
import torch
from liftoff import parse_opts

import src.estimators as estimators
import src.io_utils as ioutil
import src.rl_utils as rlutil
from src.console import rprint as print
from src.replay import OfflineExperienceReplay


class Learner:
    """ Base class for all sorts of Learners. """

    def __init__(self, estimator, optimizer, loss_fn, *args, **kwargs):
        self._estimator = estimator
        self._optimizer = optimizer
        self._loss_fn = getattr(torch.nn, loss_fn)()
        self._device = rlutil.get_estimator_device(estimator)

    def __call__(self, batch, optimize=True):
        batch = rlutil.to_device(batch, self._device)
        return self.learn(batch, optimize=optimize)

    def learn(self, batch, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `learn`"
        )


class InverseKinematicsLearner(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn(self, batch, optimize=True):
        states, actions, _, states_, done = batch
        states = states[done.flatten(), :, :, :]
        actions = actions[done.flatten(), :]

        logits = self._estimator((states, states_))
        loss = self._loss_fn(logits, actions.flatten())
        loss.backward()
        if optimize:
            self._optimizer.step()

        # print(actions.flatten().cpu().numpy())
        # print(logits.argmax(dim=-1).flatten().cpu().detach().numpy())
        # print("---")

        acc = logits.argmax(dim=-1).eq(actions.flatten()).sum()
        return loss.item(), acc.item(), states.shape[0]


def run(opt):
    """ Entry point of the program. """

    estimator = getattr(estimators, opt.estimator.name)(18, **opt.estimator.args)
    estimator.to(torch.device("cuda"))
    optimizer = getattr(torch.optim, opt.optim.name)(
        estimator.parameters(), **opt.optim.args
    )
    learner = InverseKinematicsLearner(
        estimator=estimator, optimizer=optimizer, loss_fn="CrossEntropyLoss",
    )
    print(estimator)

    N, batch_size = 1, 128

    # configure the guts of the experiment
    replay = OfflineExperienceReplay(
        opt.replay_path, True, batch_size=batch_size, device="cpu", capacity=int(1e6),
    )
    print(replay)

    # replay.reload_N(N=N, workers=N, ckpt_idxs=[42, 32])

    for epoch in range(1, 51):
        replay.reload_N(N=N, workers=N, ckpt_idxs=[0])
        acc, loss, total = [], [], 0
        for step in range(1, int((N * 1e6) / batch_size) + 1):
            batch = replay.sample(idxs=torch.randint(0, 500, (batch_size,)).tolist())
            loss_, acc_, ns_ = learner(batch)

            acc.append(acc_)
            loss.append(loss_)
            total += ns_

            if step % 500 == 0:
                correct = torch.tensor(acc).sum().item()
                print(
                    "[{:02d}/{:06d}] acc={:3.2f}/{:6d}, loss={:2.3f}".format(
                        epoch,
                        step,
                        correct / (total - 1) * 100,
                        total,
                        torch.tensor(loss).mean(),
                    )
                )
        if epoch % 10 == 0:
            ioutil.checkpoint_agent(
                opt.out_dir,
                epoch,
                estimator=estimator,
                # target_estimator=policy_evaluation.target_estimator,
                optim=optimizer,
                cfg=opt,
                replay=None,
                accuracy=correct / (total - 1) * 100,
            )

    return

    #
    #
    #
    #
    #
    #
    #
    #
    #

    # configure the logger
    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_loss", metargs=["trn_loss", 1]),
        rlog.FPSMetric("lrn_tps", metargs=["lrn_steps"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )
    # add some hardware and git info, log and save config file
    opt = ioutil.add_platform_info(opt)
    rlog.info("\n" + ioutil.config_to_string(opt))
    ioutil.save_config(opt, opt.out_dir)

    # configure the guts of the experiment
    replay = OfflineExperienceReplay(opt.replay_path, True, device=opt.replay.device)

    # start training
    steps = 0
    start_epoch = 1
    for epoch in range(start_epoch, opt.epoch_cnt + 1):

        replay.reload_N(N=opt.replay_cnt, workers=int(opt.omp_num_threads or 1))
        rlog.info(f"Sampled N={opt.replay_cnt} replay buffers.")

        # train for 250,000 steps (well, actually 250k / 4)
        total_step_cnt = train_offline_one_epoch(
            replay,
            learner,
            rlog.getRootLogger(),
            opt.train_steps,
            total_steps=total_step_cnt,
        )
        rlog.traceAndLog(epoch * opt.train_steps)

        # # validate for 125,000 steps
        # validate(
        #     AGENTS[opt.agent.name]["policy_improvement"](
        #         policy_improvement.estimator, opt.action_cnt, epsilon=opt.val_epsilon
        #     ),
        #     get_env(opt, mode="testing"),
        #     opt.valid_step_cnt,
        #     rlog.getRootLogger(),
        # )
        rlog.traceAndLog(epoch * opt.train_steps)

        # save the checkpoint
        if opt.save:
            ioutil.checkpoint_agent(
                opt.out_dir,
                steps,
                estimator=learner.estimator,
                # target_estimator=policy_evaluation.target_estimator,
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
