import unittest

import torch
import numpy as np
from sklearn.metrics import jaccard_score

from src.utils import get_ious


class TestIoU(unittest.TestCase):
    """ TestCase for the IOU function. """

    def __init__(self, *args, **kwargs):
        super(TestIoU, self).__init__(*args, **kwargs)
        self.precision = 4

    def test_one_equal(self):
        """ Identical output and target, iou should be 1.0 """
        label_num = 42
        expected_result = 1.0

        output = torch.randint(0, label_num, (1, 4, 3)).long()
        target = output.clone()
        mean_iou, _ = get_ious(output, target, range(label_num))

        self.assertEqual(expected_result, mean_iou)

    def test_two_classes(self):
        classes = [0, 1]
        target = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
        )
        output = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                ]
            ]
        )

        skl_iou = jaccard_score(
            target.reshape(-1), output.reshape(-1), average="macro"
        )
        skl_ious = jaccard_score(
            target.reshape(-1), output.reshape(-1), average=None
        )
        iou, ious = get_ious(
            torch.from_numpy(output), torch.from_numpy(target), classes
        )

        self.assertAlmostEqual(iou, skl_iou, places=self.precision)
        for iou, skl_iou in zip(list(ious.values()), list(skl_ious)):
            self.assertAlmostEqual(iou, skl_iou, places=self.precision)

    def test_three_classes(self):
        classes = [0, 1, 2]
        target = np.array(
            [
                [
                    [2, 2, 0, 0, 0],
                    [2, 2, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
        )
        output = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0],
                    [0, 2, 2, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                ]
            ]
        )

        skl_iou = jaccard_score(
            target.reshape(-1), output.reshape(-1), average="macro"
        )
        skl_ious = jaccard_score(
            target.reshape(-1), output.reshape(-1), average=None
        )
        iou, ious = get_ious(
            torch.from_numpy(output), torch.from_numpy(target), classes
        )

        self.assertAlmostEqual(iou, skl_iou, places=self.precision)
        for iou, skl_iou in zip(list(ious.values()), list(skl_ious)):
            self.assertAlmostEqual(iou, skl_iou, places=self.precision)

    def test_missing_in_target(self):
        target = np.array(
            [
                [
                    [2, 2, 0, 0, 0],
                    [2, 2, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
        )
        output = np.array(
            [
                [
                    [4, 0, 0, 0, 0],
                    [0, 2, 2, 3, 3],
                    [0, 2, 2, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                ]
            ]
        )

        skl_iou = jaccard_score(
            target.reshape(-1), output.reshape(-1), average="macro"
        )
        skl_ious = jaccard_score(
            target.reshape(-1), output.reshape(-1), average=None
        )
        iou, ious = get_ious(torch.from_numpy(output), torch.from_numpy(target))

        self.assertAlmostEqual(iou, skl_iou, places=self.precision)
        for iou, skl_iou in zip(list(ious.values()), list(skl_ious)):
            self.assertAlmostEqual(iou, skl_iou, places=self.precision)

    def test_missing_in_predicted(self):
        target = np.array(
            [
                [
                    [2, 2, 0, 3, 3],
                    [2, 2, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 4],
                    [0, 0, 0, 0, 4],
                ]
            ]
        )
        output = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0],
                    [0, 2, 2, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                ]
            ]
        )

        skl_iou = jaccard_score(
            target.reshape(-1), output.reshape(-1), average="macro"
        )
        skl_ious = jaccard_score(
            target.reshape(-1), output.reshape(-1), average=None
        )
        iou, ious = get_ious(torch.from_numpy(output), torch.from_numpy(target))

        self.assertAlmostEqual(iou, skl_iou, places=self.precision)
        for iou, skl_iou in zip(list(ious.values()), list(skl_ious)):
            self.assertAlmostEqual(iou, skl_iou, places=self.precision)

    def test_large_rand(self):
        label_num = 14  # as in VKITTI :)
        output = torch.randint(0, label_num, (1, 100, 120)).long()
        target = torch.randint(0, label_num, (1, 100, 120)).long()

        skl_iou = jaccard_score(
            target.view(-1).numpy(), output.view(-1).numpy(), average="macro"
        )
        skl_ious = jaccard_score(
            target.view(-1).numpy(), output.view(-1).numpy(), average=None
        )
        iou, ious = get_ious(output, target)

        self.assertAlmostEqual(iou, skl_iou, places=self.precision)
        for iou, skl_iou in zip(list(ious.values()), list(skl_ious)):
            self.assertAlmostEqual(iou, skl_iou, places=self.precision)

    def test_no_batch(self):
        label_num = 42
        batch_size = 5

        outputs = torch.randint(0, label_num, (batch_size, 9, 6)).long()
        targets = torch.randint(0, label_num, (batch_size, 9, 6)).long()

        with self.assertRaises(AssertionError):
            get_ious(outputs, targets)

    def test_batch(self):
        label_num = 42
        batch_size = 5
        h, w = 513, 1699

        outputs = torch.randint(0, label_num, (batch_size, h, w)).long()
        targets = torch.randint(0, label_num, (batch_size, h, w)).long()

        iou_b, _ = get_ious(
            outputs.view(1, batch_size * h, w),
            targets.view(1, batch_size * h, w),
        )

        iou_a = []
        for output, target in zip(outputs, targets):
            iou_a.append(get_ious(output.unsqueeze(0), target.unsqueeze(0))[0])

        print(torch.tensor(iou_a).mean().item(), iou_b)
