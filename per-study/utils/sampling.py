""" Various sampling strategies.
"""

import random
import itertools
import numpy as np


def get_all_transitions(env, n=3):
    """ Sample all the transitions equivalent to a random policy sampling all
        the possible sequence of actions leading to termination of the MDP.

        There are:
            - 2^N seqences of possible actions
            - 2^(N+1)-2 transitions for all the sequences
    """

    buffer = []
    actions = [0, 1]

    # Generate all possible sequence of actions
    action_sequences = list(itertools.product(*[actions] * n))
    random.shuffle(action_sequences)

    for act_seq in action_sequences:
        state, done = env.reset(), False

        for action in act_seq:
            phi_state = np.hstack([np.ones((1, 1)), state])

            state_, reward, done, _ = env.step(action)

            phi_state_ = np.hstack([np.ones((1, 1)), state_])
            buffer.append((phi_state, action, reward, phi_state_, done))

            state = state_

            if done:
                break

    seq_len = len(action_sequences)

    return buffer


def main():
    """ Entry point. """
    import gym
    import gym_fast_envs

    for n in range(2, 16):
        env = gym.make(f"BlindCliffWalk-N{n}-v0")
        get_all_transitions(env, n)


if __name__ == "__main__":
    main()
