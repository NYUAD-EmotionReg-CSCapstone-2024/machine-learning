import numpy as np
from typing import List, Set
from .DatasetSplitter import DatasetSplitter
from torch.utils.data import Subset

class KFoldSplitter(DatasetSplitter):
    """
    K-Fold splitter for datasets with overlap.
    
    This class splits the dataset into K folds for cross-validation. The user can specify 
    the fold index to use as the test set, with all remaining folds serving as the training set.
    
    Args:
        dataset (Dataset): The dataset to be split.
        k (int): The number of folds for cross-validation.
        shuffle (bool): Whether to shuffle the dataset indices before splitting. Defaults to True.
        overlap_ratio (float): The ratio of overlap between consecutive chunks.
    """
    def __init__(self, dataset, k: int, shuffle: bool = True) -> None:
        super().__init__(dataset, shuffle)
        self.k: int = k

        if self.shuffle:
            self._shuffle()
        
        self._split()
    
    def set_fold(self, fold_idx: int) -> None:
        """
        Sets the current fold index, defining which fold will be used as the test set.
        """
        if fold_idx < 0 or fold_idx >= self.k:
            raise ValueError(f"Invalid fold index: {fold_idx}. Must be between 0 and {self.k - 1}.")
        
        self.cur_fold: int = fold_idx
        self.test_indices: List[int] = self.folds[fold_idx]
        
        # Exclude test set indices from the full dataset indices to get the training set
        test_set: Set[int] = set(self.test_indices)
        self.train_indices: List[int] = [idx for idx in self.indices if idx not in test_set]


    def _shuffle(self) -> None:
        """Shuffles the indices of the dataset in place."""
        np.random.shuffle(self.indices)

    def _split(self) -> None:
        """
        Splits dataset indices into K folds.
        
        Each fold is approximately equal in size, first few folds have one extra element if the dataset
        size is not divisible by K.
        """
        self.folds: List[List[int]] = []
        fold_size: int = len(self.dataset) // self.k
        extra: int = len(self.dataset) % self.k

        start: int = 0
        for i in range(self.k):
            fold_len = fold_size + 1 if i < extra else fold_size
            self.folds.append(self.indices[start:start + fold_len])
            start += fold_len

    @property
    def trainset(self) -> Subset:
        """Returns the training set for the current fold as a Subset of the dataset."""
        if not hasattr(self, 'cur_fold'):
            raise ValueError("Please call the set_fold() method first.")
        return super().trainset

    @property
    def testset(self) -> Subset:
        """Returns the test set for the current fold as a Subset of the dataset."""
        if not hasattr(self, 'cur_fold'):
            raise ValueError("Please call the set_fold() method first.")
        return super().testset
