import numpy as np

from .DatasetSplitter import DatasetSplitter

class RandomSplitter(DatasetSplitter):
    """
    Randomly split the dataset into training and testing sets.
    Args:
        dataset (Dataset): The dataset to be split.
        train_ratio (float): The ratio of the training set.
        shuffle (bool): Whether to shuffle the dataset before splitting.
    """
    def __init__(self, dataset, train_ratio=0.8, shuffle=True):
        super().__init__(dataset, shuffle)
        self.ratio = train_ratio
        self.indices = list(range(len(dataset)))
        
        if self.shuffle:
            self._shuffle()
        
        self._split()

    def _shuffle(self):
        np.random.shuffle(self.indices)

    def _split(self):
        split_idx = int(self.ratio * len(self.dataset))
        self.train_indices = self.indices[:split_idx]
        self.test_indices = self.indices[split_idx:]