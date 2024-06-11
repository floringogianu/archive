from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T

import src.joint as J
from src.datasets import VirtualKITTI as VK


def get_palette(root):
    print("\n Palette:")
    colors, _ = [], []
    for rgb, idx in sorted(VK(root).rgb2idx.items(), key=lambda kv: kv[1]):
        # filter redundant colors for cars and vans
        if idx not in _:
            colors.append(rgb)
            _.append(idx)
            print(f"{idx:2d}) {VK.idx2label[idx]:>12} {rgb}")
    return torch.tensor(colors).numpy().astype("uint8")


def main(cmdl):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # get the data
    if cmdl.dset == "vkitti":
        print(
            "Opening VKITTI :\n  - world: {}\n  - variation: {}.\n".format(
                cmdl.world, cmdl.variation
            )
        )
        root = Path("./data/vkitti")
        img_path = root / "vkitti_1.3.1_rgb" / cmdl.world / cmdl.variation
        tgt_path = root / "vkitti_1.3.1_scenegt" / cmdl.world / cmdl.variation
    elif cmdl.dset == "kitti":
        print("Opening real KITTI images")
        root = Path("./data/rkitti/data_semantics/training")
        img_path = root / "image_2"
        tgt_path = root / "semantic_rgb"

    img_paths = list(img_path.glob("**/*.png"))
    tgt_paths = list(tgt_path.glob("**/*.png"))

    # configure the model and load the checkpoint
    model = torch.hub.load("pytorch/vision", "deeplabv3_resnet101")
    # monkey-patch the DeepLab-v3 classifier: 21 classes (COCO) to 14 (VKitti)
    model.classifier[4] = torch.nn.Conv2d(256, 14, 1)
    print("Loading state...")
    model.load_state_dict(torch.load(cmdl.model_path)["model_state"])
    model.to(device)
    model.eval()

    # Define transforms
    joint_transform = J.Compose(
        [J.RandomResize(min_size=525, max_size=750), J.RandomCrop(513)]
    )
    img_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create a color palette
    palette = get_palette("./data/vkitti")

    print("\nLoading samples...")
    for img_path, tgt_path in zip(img_paths, tgt_paths):
        print(img_path, tgt_path)

        image = Image.open(img_path.as_posix())
        target = Image.open(tgt_path.as_posix())

        input_image, ground_truth = joint_transform(image, target)
        input_tensor = img_transform(input_image)
        input_batch = input_tensor.unsqueeze(0)  # add minibatch dim

        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = model(input_batch)["out"][0]
        output_predictions = output.argmax(0)

        # make image out of semantic segmentation labels
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(
            input_image.size
        )
        r.putpalette(palette)
        r = r.convert("RGB")

        # plot
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(input_image)
        fig.add_subplot(1, 3, 2)
        plt.imshow(ground_truth)
        fig.add_subplot(1, 3, 3)
        plt.imshow(r)

        # r.save("test_segment.png")
        plt.show()


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(description="ShapeBias")
    PARSER.add_argument("model_path", type=str, help="Path of the model.")
    PARSER.add_argument(
        "-d",
        "--dset",
        type=str,
        default="vkitti",
        help="Dataset to load: `vkitti`, `kitti`.",
    )
    PARSER.add_argument(
        "-w",
        "--world",
        type=str,
        default="0001",
        help="Virtual Kitti world. Options are 0001, 0002, 0006, 0018, 0020.",
    )
    PARSER.add_argument(
        "-v",
        "--variation",
        type=str,
        default="clone",
        help="Virtual Kitti variation. Otps are clone, sunset, fog, rain, ...",
    )
    main(PARSER.parse_args())
