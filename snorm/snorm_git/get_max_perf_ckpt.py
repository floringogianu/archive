""" Finds the best checkpoints and saves their index (step) on disk.
"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.nb_utils import IGNORE, get_data


ROOT_PATH = "./results/2020Nov14-202129_minatar_dqn_size/"


def get_fp(results_path):
    """Returns a list of lists, each containing
    the results file_paths of a trial."""
    file_paths = defaultdict(list)
    for root, _, files in os.walk(results_path):
        if ".__leaf" in files and "__crash" not in files:
            for file in files:
                if file not in IGNORE:
                    file_paths[Path(root)].append(file)
    return file_paths


def main():
    """Entry point"""

    fn_paths = {
        k: [f for f in v if "--" not in f] for k, v in get_fp(ROOT_PATH).items()
    }
    df = get_data(
        fn_paths,
        ["game"],
        pkl_metrics=["val_R_ep"],
        cb=lambda p, cfg, data: [("path", p)],
        cfg_fname="post_cfg.yml",
    )

    print(df.sample(10))
    print(df.describe())

    df_max_perf = df.loc[df.groupby(["path"])["val_R_ep"].idxmax()][
        ["path", "step"]
    ].reset_index(drop=True)

    print("Writing max_ckpt indices...")
    for _, row in df_max_perf.iterrows():
        with open(f"{row.path}/max_ckpt", "w") as f:
            f.write(str(row.step))

    idx_fps = list(Path(ROOT_PATH).glob("*/**/max_ckpt"))
    sampled = np.random.choice(idx_fps, 10)
    for p in sampled:
        with open(p, "r") as f:
            print(int(f.readline()), "  -->   ", p)


if __name__ == "__main__":
    main()
