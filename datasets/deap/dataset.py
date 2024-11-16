import os
import pickle
from torch.utils.data import Dataset

# Configuration constants
SAMPLE_RATE = 128
TOTAL_PARTICIPANTS = 32
NUM_EEG_CHANNELS = 32
NUM_LABELS = 3

class DEAPDataset(Dataset):
    """
    A PyTorch Dataset to load and preprocess the DEAP dataset.

    Args:
        root_dir (str): Root directory of the DEAP dataset.
        participants (list[int], optional): List of participant IDs (default is all participants).
        chunk_duration (int, optional): Duration of data chunks (in seconds, default is 1).
        overlap (float, optional): Overlap between chunks as a fraction (default is 0).
    """

    def __init__(self, root_dir, participants=None, chunk_duration=1, overlap=0):
        super().__init__()
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data_preprocessed_python")
        self.participants = participants or list(range(1, TOTAL_PARTICIPANTS + 1))
        self.chunk_duration = chunk_duration
        self.overlap = overlap

        self.data = []  # Stores EEG data chunks
        self.labels = []  # Stores labels for chunks

        self._validate_params()
        self._load_data()

    def _validate_params(self):
        """Validates input parameters and dataset structure."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory '{self.root_dir}' not found.")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found.")

        for participant in self.participants:
            if not (1 <= participant <= TOTAL_PARTICIPANTS):
                raise ValueError(f"Participant ID {participant} is not valid.")
            data_file = os.path.join(self.data_dir, f"s{str(participant).zfill(2)}.dat")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Data file '{data_file}' for participant {participant} not found.")

    def _load_file(self, file):
        """Loads a participant's data file."""
        file_path = os.path.join(self.data_dir, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            return data["data"], data["labels"]

    def _load_data(self):
        """Loads and processes all data for the specified participants."""
        chunk_length = SAMPLE_RATE * self.chunk_duration
        overlap_length = int(chunk_length * self.overlap)

        for participant in self.participants:
            data, labels = self._load_file(f"s{str(participant).zfill(2)}.dat")

            for trial_idx in range(data.shape[0]):  # Process each trial
                trial_data = data[trial_idx]  # EEG data: (channels, time)
                trial_labels = labels[trial_idx]  # Trial labels

                for start in range(0, trial_data.shape[1] - chunk_length + 1, chunk_length - overlap_length):
                    end = start + chunk_length
                    self.data.append(trial_data[:, start:end])  # Add chunk
                    self.labels.append(trial_labels)  # Add labels

    def __len__(self):
        """Returns the total number of chunks in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves (chunk, label) where chunk is EEG data"""
        return self.data[idx][:NUM_EEG_CHANNELS], self.labels[idx][:NUM_LABELS]