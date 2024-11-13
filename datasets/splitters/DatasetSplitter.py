from abc import ABC, abstractmethod
from typing import List, Optional
from torch.utils.data import Dataset, Subset

class DatasetSplitter(ABC):
    """
    Abstract class for dataset splitters.
    
    Provides a base class for creating different dataset splitters, enforcing that any
    derived class implements the `_split` method.
    
    Args:
        dataset (Dataset): The dataset to be split.
        shuffle (bool): Whether to shuffle the dataset indices before splitting.
    """
    def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
        self.dataset: Dataset = dataset
        self.shuffle: bool = shuffle
        self.indices: List[int] = list(range(len(dataset)))
        self.train_indices: Optional[List[int]] = None
        self.test_indices: Optional[List[int]] = None

    @abstractmethod
    def _split(self) -> None:
        """Method to split dataset indices. Must be implemented by subclasses."""
        pass

    @property
    def trainset(self) -> Subset:
        """
        Returns the training set as a Subset of the dataset.
        
        Raises:
            ValueError: If the training indices are not set, indicating that the split 
                        method has not been called.
        """
        if self.train_indices is None:
            raise ValueError("Please call the split() method first.")
        return Subset(self.dataset, self.train_indices)
    
    @property
    def testset(self) -> Subset:
        """
        Returns the test set as a Subset of the dataset.
        
        Raises:
            ValueError: If the test indices are not set, indicating that the split 
                        method has not been called.
        """
        if self.test_indices is None:
            raise ValueError("Please call the split() method first.")
        return Subset(self.dataset, self.test_indices)