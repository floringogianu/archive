# DO(VAE)

## Installation

This repo uses a submodule, therefore please clone it with:

```sh
git clone git@github.com:floringogianu/do-vae.git --recursive
```

If you already cloned without the `--recursive` flag then:

```sh
git submodule init
git submodule update
```

Finally, install the conda env:

```sh
conda env create -f environment.yml
```