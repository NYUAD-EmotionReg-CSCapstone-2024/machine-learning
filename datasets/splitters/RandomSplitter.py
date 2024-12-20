import numpy as np
from .DatasetSplitter import DatasetSplitter

class RandomSplitter(DatasetSplitter):
    """
    Randomly split the dataset into training and testing sets.

    Args:
        dataset (Dataset): The dataset to be split.
        train_ratio (float): The ratio of the training set.
        shuffle (bool): Whether to shuffle the dataset before splitting.
        overlap_ratio (float): The ratio of overlap between consecutive chunks.
    """
    def __init__(self, dataset, train_ratio=0.8, shuffle=True, overlap_ratio=0.5):
        super().__init__(dataset, shuffle)
        self.ratio = train_ratio
        self.overlap_ratio = overlap_ratio  # New parameter for overlap
        self.indices = list(range(len(dataset)))

        if self.shuffle:
            self._shuffle()

        self._split()

    def _shuffle(self):
        np.random.shuffle(self.indices)

    def _split(self):
        """
        Splits the dataset into training and testing sets and applies overlap to each split.
        """
        split_idx = int(self.ratio * len(self.dataset))

        # Split into non-overlapping base segments
        train_base = [self.dataset.segments[i] for i in self.indices[:split_idx]]
        test_base = [self.dataset.segments[i] for i in self.indices[split_idx:]]

        # Generate overlapping chunks and map them back to integer indices
        self.train_indices = self._map_segments_to_indices(
            self.generate_overlapping_chunks(train_base, self.overlap_ratio)
        )
        self.test_indices = self._map_segments_to_indices(
            self.generate_overlapping_chunks(test_base, self.overlap_ratio)
        )

    def _map_segments_to_indices(self, overlapping_segments):
        """
        Maps overlapping segments back to integer indices in the dataset.

        Args:
            overlapping_segments (list): List of overlapping segments (dictionaries).

        Returns:
            list: List of integer indices corresponding to the dataset.
        """
        segment_to_index = {segment["data_id"]: idx for idx, segment in enumerate(self.dataset.segments)}
        return [segment_to_index[seg["data_id"]] for seg in overlapping_segments]
