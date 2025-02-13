from typing import List, Union, Set
from torch.utils.data.dataset import Dataset
from .DatasetSplitter import DatasetSplitter

class LNSOSplitter(DatasetSplitter):
    """
    Leave-N Participants Out Splitter (LNSO) with overlap.
    
    This splitter splits the dataset by leaving out a certain number of participants for 
    testing, while using the remaining participants for training.
    
    Args:
        dataset (Dataset): The dataset to be split.
        num_participants (Union[int, List[int]]): The number of participants to be used for testing.
                                                   Can be an integer specifying how many participants to leave out, 
                                                   or a list of participant IDs.
        shuffle (bool): Whether to shuffle the dataset indices before splitting. Defaults to True.
        overlap_ratio (float): The ratio of overlap between consecutive chunks.
    """
    def __init__(self, dataset: Dataset, num_participants: Union[int, List[int]]) -> None:
        super().__init__(dataset)
        
        self.participants: List[str] = dataset.participants
        self.data_ids: List[str] = dataset.data_ids
        self.num_participants: int = num_participants

        # Determine the number of participants to leave out
        if isinstance(self.num_participants, List):
            # Check if all participants are valid
            for pid in self.num_participants:
                if pid not in self.participants:
                    raise ValueError(f"Invalid participant ID: {pid}. Participant not found in the dataset.")
            self.test_participants: Set[str] = set(self.num_participants)
            self.train_participants: Set[str] = set(self.participants) - self.test_participants
        else:
            if self.num_participants < 1 or self.num_participants >= len(self.participants):
                raise ValueError(f"Invalid number of participants: {self.num_participants} : Must be between 1 and {len(self.participants)-1}.")
            self.test_participants: Set[str] = set(self.participants[:self.num_participants])
            self.train_participants: Set[str] = set(self.participants[self.num_participants:])
        
        # Split the data based on the participants
        self._split()

    def _split(self) -> None:
        """Splits dataset into training and testing sets based on participants."""  
        self.train_indices: List[int] = []
        self.test_indices: List[int] = []

        for idx, data_id in enumerate(self.data_ids):
            # Extract participant ID from the data_id
            pid = data_id.split("_")[0]

            # Assign to train or test set based on participant ID
            if int(pid) in self.train_participants or pid in self.train_participants:
                self.train_indices.append(idx)
            else:
                self.test_indices.append(idx)