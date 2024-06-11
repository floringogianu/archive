""" Utils related to input/output."""
import shutil
from argparse import Namespace
from gzip import GzipFile
from pathlib import Path

import rlog
import torch
import yaml
from termcolor import colored as clr


def config_to_string(
    cfg: Namespace,
    indent: int = 0,
    color: bool = True,
    verbose: bool = False,
    newl: bool = False,
) -> str:
    """Creates a multi-line string with the contents of @cfg."""

    text = "\n" if newl else ""
    for key, value in cfg.__dict__.items():
        if key.startswith("__") and not verbose:
            # censor some fields
            pass
        else:
            ckey = clr(key, "yellow", attrs=["bold"]) if color else key
            text += " " * indent + ckey + ": "
            if isinstance(value, Namespace):
                text += "\n" + config_to_string(value, indent + 2, color=color)
            else:
                cvalue = clr(str(value), "white") if color else str(value)
                text += cvalue + "\n"
    return text


def namespace_to_dict(namespace: Namespace) -> dict:
    """Deep (recursive) transform from Namespace to dict"""
    dct: dict = {}
    for key, value in namespace.__dict__.items():
        if isinstance(value, Namespace):
            dct[key] = namespace_to_dict(value)
        else:
            dct[key] = value
    return dct


def dict_to_namespace(dct: dict) -> Namespace:
    """Deep (recursive) transform from dict to Namespace"""
    namespace = Namespace()
    for key, value in dct.items():
        name = key.rstrip("_")
        if isinstance(value, dict) and not key.endswith("_"):
            setattr(namespace, name, dict_to_namespace(value))
        else:
            setattr(namespace, name, value)
    return namespace


def _sanitize_dict(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            _sanitize_dict(v)
        else:
            if not isinstance(v, (int, float, str, list, tuple)):
                cfg[k] = str(v)


def save_config(cfg, path):
    """Save namespace or dict to disk."""
    if isinstance(cfg, Namespace):
        cfg = namespace_to_dict(cfg)
    elif isinstance(cfg, dict):
        pass
    else:
        raise TypeError(f"Don't know what to do with cfg of type {type(cfg)}.")

    # who knows what I'm storing in there so...
    _sanitize_dict(cfg)

    with open(Path(path) / "post_cfg.yml", "w") as outfile:
        yaml.safe_dump(cfg, outfile, default_flow_style=False)


def read_config(cfg_path, info=True):
    """Read a config file and return a namespace."""
    with open(cfg_path) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
    return dict_to_namespace(config_data)


def checkpoint_agent(path, crt_step, **kwargs):
    save_every_replay = kwargs.get("save_every_replay", False)

    # save checkpoint
    to_save = {"step": crt_step}
    replay_path = None
    for k, v in kwargs.items():
        if k == "replay" and v is not None:
            replay_path = v.save(path, crt_step, save_all=save_every_replay)
        elif isinstance(v, (torch.nn.Module, torch.optim.Optimizer)):
            to_save[f"{k}_state"] = v.state_dict()
        elif isinstance(v, (Namespace)):
            to_save[k] = namespace_to_dict(v)
        else:
            to_save[k] = v

    if replay_path is not None:
        # save checkpoints only when saving the replay
        with open(f"{path}/checkpoint.gz", "wb") as f:
            with GzipFile(fileobj=f) as outfile:
                torch.save(to_save, outfile)

        # when saving every replay the replay file name is replay_xxx.gz
        if save_every_replay:
            shutil.copyfile(replay_path, Path(path) / "replay.gz")

        # sometimes saving the replay fails and we end up with a bad `replay.gz`.
        # therefore we make sure we have at least one good copy of the previous replay.
        shutil.copyfile(replay_path, Path(path) / "prev_replay.gz")
        # same for the checkpoint
        shutil.copyfile(Path(path) / "checkpoint.gz", Path(path) / "prev_checkpoint.gz")

    # save every model
    with open(f"{path}/model_{crt_step:08d}.gz", "wb") as f:
        with GzipFile(fileobj=f, mode="w") as outfile:
            torch.save(
                {k: v for k, v in to_save.items() if k in ["step", "estimator_state"]},
                outfile,
            )

    rlog.info("Saved the agent's state.")
