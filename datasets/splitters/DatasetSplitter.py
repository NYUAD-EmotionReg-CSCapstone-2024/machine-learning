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
    
    def generate_overlapping_chunks(self, base_segments, overlap_ratio=0.5):
        """
        Generate overlapping chunks for a list of base segments.
        Args:
            base_segments (list): List of segment metadata dictionaries.
            overlap_ratio (float): The ratio of overlap between consecutive chunks.

        Returns:
            list: Overlapping segments with updated start/end indices.
        """

        if not (0 <= overlap_ratio <= 1):
            raise ValueError("Overlap must be between 0 and 1")

        # Map the dictionary back into the appropriate index for training
        segment_to_index = {segment["data_id"]: idx for idx, segment in enumerate(self.dataset.segments)}
        overlapping_chunks = []
        for segment in base_segments:
            start = segment["start"]
            end = segment["end"]
            segment_length = end - start
            overlap_length = int(segment_length * overlap_ratio)

            # Generate overlapping segments
            for i in range(start, end - segment_length + 1, max(segment_length - overlap_length, 1)):
                overlapped_segment = segment.copy()
                overlapped_segment["start"] = i
                overlapped_segment["end"] = i + segment_length
                overlapping_chunks.append(segment_to_index[overlapped_segment["data_id"]])

        return overlapping_chunks
