""" Entry point. """
import rlog
import torch
from liftoff import parse_opts
from torch.utils.data import DataLoader

import src.estimators as estimators
import src.io_utils as ioutil
import src.rl_utils as rlutil
from src.console import rprint as print
from src.dataset import Atari, collate


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
        # states = states[done.flatten(), :, :, :]
        # actions = actions[done.flatten(), :]

        logits = self._estimator((states, states_))
        loss = self._loss_fn(logits, actions.flatten())
        loss.backward()
        if optimize:
            self._optimizer.step()

        # print(actions.flatten().cpu().numpy())
        # print(logits.argmax(dim=-1).flatten().cpu().detach().numpy())
        # print("---")

        acc = logits.argmax(dim=-1).eq(actions.flatten()).sum()
        return loss.item(), acc.item()


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

    batch_size = 128
    trn_dset = Atari(opt.data_path)
    trn_loader = DataLoader(
        trn_dset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )
    print("Length of Atari: ", len(trn_dset))

    for epoch in range(1, 10):
        acc, loss = [], []
        for step, batch in enumerate(trn_loader):
            loss_, acc_, = learner(batch)

            acc.append(acc_)
            loss.append(loss_)

            if (step != 0) and (step % 500 == 0):
                correct = torch.tensor(acc).sum().item()
                print(
                    "[{:02d}/{:08d}] acc={:6.2f} ({:8d}/{:8d}), loss={:2.3f}".format(
                        epoch,
                        step,
                        correct / (len(acc) * batch_size) * 100,
                        correct,
                        len(acc) * batch_size,
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


def main():
    """ Liftoff
    """
    run(parse_opts())


if __name__ == "__main__":
    main()
