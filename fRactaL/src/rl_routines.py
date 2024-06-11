""" RL-Routines
"""
import torch


class Episode:
    """ An iterator accepting an environment and a policy, that returns
    experience tuples.
    """

    def __init__(self, env, policy, _state=None):
        self.env = env
        self.policy = policy
        self.__R = 0.0  # TODO: the return should also be passed with _state
        self.__step_cnt = -1
        if _state is None:
            self.__state, self.__done = self.env.reset(), False
        else:
            self.__state, self.__done = _state, False

    def __iter__(self):
        return self

    def __next__(self):
        if self.__done:
            raise StopIteration

        _pi = self.policy.act(self.__state)
        _state = self.__state.clone()
        self.__state, reward, self.__done, info = self.env.step(_pi.action)

        self.__R += reward
        self.__step_cnt += 1
        return (_state, _pi, reward, self.__state, self.__done, info)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print("Episode done")

    @property
    def Rt(self):
        """ Return the expected return."""
        return self.__R

    @property
    def steps(self):
        """ Return steps taken in the environment.
        """
        return self.__step_cnt


def train_rounds(steps, interval):
    """ Returns a generator of tuples making a training round.

    Args:
        steps (int): Total number of training steps.
        interval (int): Frequency of training rounds.

    Returns:
        generator: Generator of (start, end) tuples.

    Example:
        steps, interval = 1_000_000, 5_000
        val_freq = 5_000

        [(0, 5000), (5000, 10000), ...]
    """
    return ((i * interval, (i * interval) + interval) for i in range(steps // interval))


def train_one_epoch(
    env,
    agent,
    epoch_step_cnt,
    update_freq,
    target_update_freq,
    logger,
    total_steps=0,
    last_state=None,
):
    """ Policy iteration for a given number of steps. """

    replay, policy, policy_evaluation = agent
    policy_evaluation.estimator.train()
    policy_evaluation.target_estimator.train()

    while True:
        # do policy improvement steps for the length of an episode
        # if _state is not None then the environment resumes from where
        # this function returned.
        for transition in Episode(env, policy, _state=last_state):

            _state, _pi, reward, state, done, _ = transition
            total_steps += 1

            # push one transition to experience replay
            replay.push((_state, _pi.action, reward, state, done))

            # learn if a minimum no of transitions have been pushed in Replay
            if replay.is_ready:
                if total_steps % update_freq == 0:
                    # sample from replay and do a policy evaluation step
                    batch = replay.sample()

                    # compute the loss and optimize
                    loss = policy_evaluation(batch)

                    # stats
                    logger.put(
                        trn_loss=loss.loss.detach().mean().item(),
                        lrn_steps=batch[0].shape[0],
                    )
                    if hasattr(loss, "entropy"):
                        logger.put(trn_entropy=loss.entropy.detach().item())

                if total_steps % target_update_freq == 0:
                    policy_evaluation.update_target_estimator()

            # some more stats
            logger.put(trn_reward=reward, trn_done=done, trn_steps=1)
            if (policy.estimator.spectral is not None) and (total_steps % 1000 == 0):
                logger.put(**policy.estimator.get_spectral_norms())
            if total_steps % 50_000 == 0:
                msg = "[{0:6d}] R/ep={trn_R_ep:2.2f}, tps={trn_tps:2.2f}"
                logger.info(msg.format(total_steps, **logger.summarize()))

            # exit if done
            if total_steps % epoch_step_cnt == 0:
                return total_steps, _state

        # This is important! It tells Episode(...) not to attempt to resume
        # an episode intrerupted when this function exited last time.
        last_state = None


def train_offline_one_epoch(
    agent, epoch_step_cnt, update_freq, target_update_freq, logger, total_steps=0,
):
    """ Policy iteration for a given number of steps. """

    replay, policy, policy_evaluation = agent
    policy_evaluation.estimator.train()
    policy_evaluation.target_estimator.train()

    while True:

        total_steps += 1

        if total_steps % update_freq == 0:
            # sample from replay and do a policy evaluation step
            batch = replay.sample()

            # compute the loss and optimize
            loss = policy_evaluation(batch)

            # stats
            logger.put(
                trn_loss=loss.loss.detach().mean().item(), lrn_steps=batch[0].shape[0],
            )
            if hasattr(loss, "entropy"):
                logger.put(trn_entropy=loss.entropy.detach().item())

        if total_steps % target_update_freq == 0:
            policy_evaluation.update_target_estimator()

        # some more stats
        if (policy.estimator.spectral is not None) and (total_steps % 1000 == 0):
            logger.put(**policy.estimator.get_spectral_norms())
        if total_steps % 50_000 == 0:
            msg = "[{0:6d}] trn_loss={trn_loss:2.4f}, tps={lrn_tps:2.2f}"
            logger.info(msg.format(total_steps, **logger.summarize()))

        # exit if done
        if total_steps % epoch_step_cnt == 0:
            return total_steps


def validate(policy, env, steps, logger):
    """ Validation routine """
    policy.estimator.eval()

    done_eval, step_cnt = False, 0
    with torch.no_grad():
        while not done_eval:
            for _, _, reward, _, done, _ in Episode(env, policy):
                logger.put(reward=reward, done=done, val_frames=1)
                step_cnt += 1
                if step_cnt >= steps:
                    done_eval = True
                    break
    env.close()
