# Do BNNs suffer from warm-up?

## Installation

Until we test that creating conda envs from `environment.yaml` works properly
I will leave here installation instructions for non-obvious packages.

```sh
conda install termcolor
pip install git+git://github.com/tudor-berariu/liftoff.git#egg=liftoff
```

## Usage

For launching **one** experiment using `liftoff`:

```sh
liftoff main.py -c ./configs/dev.yaml
```

For launching **ten** identical experiments using `liftoff`:

```sh
## generate ten config files from your dev.yaml
liftoff-prepare configs/dev.yaml --runs-no 10
## the output is just a dry-run, append `--do` to the command
liftoff-prepare configs/dev.yaml --runs-no 10 --do
## now you can launch the experiments
liftoff main.py results/some_newly_created_folder_in_results
```
