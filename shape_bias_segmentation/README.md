# Shape-biased segmentation

Repo containing code for doing domain-adaption on a segmentation task using
style transfer.


## Installation

```sh
conda env create -f environment.yaml
```

I had this fail once because of some different `numpy` version requirements
between `opencv` and `pytorch`. If that happens try:

```sh
conda env create -f environment_.yaml
```

which will install everything except opencv, followed by `conda install opencv`.


## Training DeepLab-v3/Resnet-101 on VirtualKitti

Download scripts are not yet available. Supposing you downloaded and untarred
the images and the targets in `./data/vkitti`, you can now run:

```sh
python train.py
```

This will default to using `./configs/vkitti_deeplab_v3.yml` config file which
stores most of the settings used for training.


## TODO:

- [ ] Integrate TensorBoard (maybe via [fgogianu/rlog](https://github.com/floringogianu/rlog))
- [ ] Integrate Neptune
- [ ] Add support for resuming experiments
- [ ] Decide on the evaluation routine (Small Virtual Kitti split or Real Kitti validation split)

