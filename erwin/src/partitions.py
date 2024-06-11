from torch.utils.data import Dataset, random_split
from math import floor

class PartitionedDataset:
    def __init__(self, dataset: Dataset, partition_size: float):
        if not 0 < partition_size < 1:
            raise ValueError(f"partition_size should be between 0 and 1.")
        self.dataset = dataset
        self.split_size = floor(partition_size * len(dataset))

        self.partitions = random_split(
            dataset, [self.split_size, len(dataset) - self.split_size]
        )

    def __getitem__(self, key):
        return self.partitions[key]

    def __len__(self):
        return [len(x) for x in self.partitions]