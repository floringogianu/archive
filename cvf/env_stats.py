import numpy as np
from src.wrappers import get_wrapped_atari


def main():

    for wrapper in ["dopamine"]:

        print(f"{wrapper} wrapper:")
        for mode in ["testing"]:
            env = get_wrapped_atari(
                "asterix", mode=mode, seed=99, dopamine=(wrapper == "dopamine")
            )
            actions = env.action_space.n
            print(actions)

            steps, ep, ep_rw = [0], 0, [0]

            env.reset()
            for _ in range(50_000):
                obs, reward, done, _ = env.step(np.random.randint(0, actions))

                print(obs.shape)

                steps[ep] += 1
                ep_rw[ep] += reward

                if done:
                    env.reset()
                    ep += 1
                    steps.append(0)
                    ep_rw.append(0)

            print(
                "\t{:10}: eps={:03d}, rw/ep={:2.2f}, steps/ep={:2.2f}".format(
                    mode, ep, np.mean(ep_rw), np.mean(steps)
                )
            )


if __name__ == "__main__":
    main()
