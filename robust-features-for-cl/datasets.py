import csv
import os
import sys
from torchvision.datasets import ImageFolder


def get_cl_dset(fp):
    cl_dset = {}
    with open(fp, "r") as f:
        cls_no, misc_cls, task_no = list(map(int, f.readline().split(",")))
        for line in f.readlines():
            wnids = [wnid.strip() for wnid in line.split(",")]
            supercls = wnids.pop(0)
            cl_dset[supercls] = wnids
    cl_dset["meta"] = {"cls_no": cls_no, "task_no": task_no, "misc": misc_cls}
    return cl_dset


def get_task(cl_meta_path, task_id, verbose=True):
    wnid2words = {
        r[0]: r[1]
        for r in csv.reader(
            open("data/tiny-imagenet-200/words.txt", "r"), delimiter="\t"
        )
    }

    cl_dset = get_cl_dset(cl_meta_path)
    task = [v[task_id] for k, v in cl_dset.items() if k != "meta"]
    for i, cls_name in enumerate(task):
        print(i, cls_name, " ->  ", wnid2words[cls_name])
    return task


class TinyImagenetTask(ImageFolder):
    def __init__(self, root, subset, **kwargs):
        self._subset = subset
        super(TinyImagenetTask, self).__init__(root, **kwargs)

    def _find_classes(self, dir):
        """ Finds the class folders in a dataset and filters out classes not
        appearing in the `subset`.

        Args:
            dir (string): Root directory path.
            subset (list): List of folders that make this particular dataset.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to
                (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        all_classes = [
            d
            for d in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, d))
        ]
        for cls_name in self._subset:
            assert cls_name in all_classes, f"{cls_name} not in the root path."

        # `subset` ordering is the single source of truth
        # class idxs need to be consistent across subsets
        classes = [d for d in self._subset]

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for k, v in class_to_idx.items():
            print(k, v)
        return classes, class_to_idx


def main():
    task = get_task("./cl_t8_c12.txt", 0, verbose=True)
    dset = TinyImagenetTask("./data/tiny-imagenet-200/train", task)


if __name__ == "__main__":
    main()

