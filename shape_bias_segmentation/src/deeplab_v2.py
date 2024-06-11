""" Implementation of DeepLab-v2 https://arxiv.org/pdf/1606.00915.pdf
    We are following Fig.7 and Table 3 in the paper.
"""
import torch
from torch import nn


from torchvision import models


class ASPP(nn.Module):
    """ Implements an ASPP module.
    """

    def __init__(self, in_ch, out_ch, rates):
        """ ASPP constructor.

        in_ch: no of input channels
        out_ch: no of classes
        rates: list of atrous rates (four different rates in the paper).

        Each ASPP consists of four branches, each with an Atrous Convolution
        with a different atrous rate, followed by 1x1 convolutions.
        """
        super(ASPP, self).__init__()

        for i, rate in enumerate(rates):
            self.add_module(
                "branch{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        in_ch,
                        1024,
                        kernel_size=3,
                        stride=1,
                        padding=rate,
                        dilation=rate,
                    ),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv2d(1024, out_ch, kernel_size=1, stride=1),
                ),
            )

    def forward(self, x):
        return sum([branch(x) for branch in self.children()])


class DeepLabV2(nn.Module):
    def __init__(
        self,
        class_no,
        pretrained=False,
        pretrained_backbone=True,
        backbone="vgg16",
        rates=[6, 12, 18, 24],
    ):
        """ DeepLabV2 constructor. """
        super(DeepLabV2, self).__init__()
        assert not pretrained, "Pretrained weights not available."

        if backbone in ["vgg16"]:
            aspp_in_ch = 512
        else:
            raise NotImplementedError(
                f"This DeepLab-v2 does not support {backbone} backbone."
            )

        factory = getattr(models, backbone)
        self.backbone = factory(pretrained=pretrained_backbone).features

        # Changing the second to last and last max pool layers
        max_pool_indices = [23, 30]
        for layer_idx in max_pool_indices:
            self.backbone[layer_idx].stride = 1
            self.backbone[layer_idx].padding = 1
            self.backbone[layer_idx].kernel_size = 3

        # Changing all the conv layers after the second to last max pool layer.
        conv_indices = [24, 26, 28]
        for layer_idx in conv_indices:
            self.backbone[layer_idx].dilation = (2, 2)
            self.backbone[layer_idx].padding = (2, 2)

        self.aspp = ASPP(aspp_in_ch, class_no, rates)

    def forward(self, x):
        features = self.backbone(x)
        return self.aspp(features)


def main():
    model = DeepLabV2(12, backbone="vgg16")
    print(model)



if __name__ == "__main__":
    main()
