import logging
import os
import torch
import numpy as np
import h5py as h5
from torch.utils.data.dataset import Dataset
from .channel_mappings import _channel_info as channel_mappings

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

_total_participants = 16
_total_sessions = 3
_total_emotions = 5

# Exhaustive list of participants, sessions, emotions, and channels
_participants = [str(i) for i in range(1, _total_participants + 1)]
_sessions = [str(i) for i in range(1, _total_sessions + 1)]
_emotions = [str(emotion) for emotion in range(_total_emotions)]
_channels = [str(key) for key in channel_mappings.keys()]

class SeedVDataset(Dataset):
    def __init__(self, root_dir, h5file, transform=None, participants=_participants, sessions=_sessions, emotions=_emotions, channels=_channels, load=False):
        """
        Initialize the dataset.
        """
        logging.info(f"Initializing SeedVDataset with h5file: {h5file}, load: {load}")
        self.root_dir = root_dir
        self.h5file = h5.File(os.path.join(root_dir, h5file), "r")
        self.transform = transform
        self.data_ids = []

        self.participants = participants
        self.sessions = sessions
        self.emotions = emotions
        self.channels = channels

        self._validate_params()
        self._collect_data_ids()

        self.channel_ids = [channel_mappings[channel]["index"] for channel in self.channels]

        self.load = load
        if load:
            logging.info("Loading dataset into memory...")
            self._load_in_memory()
            logging.info(f"Loaded {len(self.data)} data samples into memory successfully.")

    def _validate_params(self):
        """
        Validate the parameters passed to the dataset. All provided participants, sessions, and emotions must be present in the dataset.
        Validate the name of channels.
        """
        logging.debug("Validating parameters...")
        for pid in self.participants:
            if str(pid) not in self.h5file:
                raise ValueError(f"Participant {pid} not found in the dataset.")
            for sid in self.sessions:
                # Escape participant 7 session 1 (not working yet, need to fix)
                if str(pid) == "7" and str(sid) == "1":
                    continue
                if str(sid) not in self.h5file[str(pid)]:
                    raise ValueError(f"Session {sid} not found for participant {pid}.")
                for emotion in self.emotions:
                    emotion = str(emotion)
                    if emotion not in self.h5file[str(pid)][str(sid)]:
                        raise ValueError(f"Emotion {emotion} not found for participant {pid} in session {sid}.")
        for channel in self.channels:
            if channel not in channel_mappings:
                raise ValueError(f"Channel {channel} not found in the channel mappings.")
        logging.debug("Parameter validation successful.")

    def _collect_data_ids(self):
        """
        Collect data IDs for the dataset.
        """
        logging.debug("Collecting data IDs...")
        for pid in self.participants:
            for sid in self.sessions:
                # Escape participant 7 session 1 (not working yet, need to fix)
                if str(pid) == "7" and str(sid) == "1":
                    continue
                for emotion in self.emotions:
                    data_ids = list(self.h5file[str(pid)][str(sid)][str(emotion)].keys())
                    self.data_ids.extend(data_ids)
        np.random.shuffle(self.data_ids)
        logging.info(f"Collected {len(self.data_ids)} data IDs.")

    def _load_in_memory(self):
        """
        Load the dataset into memory.
        """
        self.data = []
        self.labels = []
        for idx, data_id in enumerate(self.data_ids):
            if idx % 100 == 0:
                logging.debug(f"Loading data sample {idx}/{len(self.data_ids)} into memory...")
            pid, sid, emotion, _ = data_id.split("_")
            chunk = self.h5file[pid][sid][emotion][data_id][()]
            chunk = chunk[self.channel_ids]
            self.data.append(chunk)
            self.labels.append(int(emotion))
        logging.info("All data loaded into memory.")

    def __len__(self):
        """
        Return the number of data samples.
        """
        return len(self.data_ids)

    def _get_from_memory(self, idx):
        """
        Fetch a sample from memory.
        """
        chunk = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return chunk, label

    def _get_from_h5(self, idx):
        """
        Fetch a sample from the HDF5 file.
        """
        data_id = self.data_ids[idx]
        pid, sid, emotion, _ = data_id.split("_")
        chunk = self.h5file[pid][sid][emotion][data_id][()]
        chunk = chunk[self.channel_ids]
        chunk = torch.tensor(chunk, dtype=torch.float32)
        label = torch.tensor(int(emotion), dtype=torch.long)
        return chunk, label

    def __getitem__(self, idx):
        """
        Fetch a data sample by index.
        """
        if self.load:
            chunk, label = self._get_from_memory(idx)
        else:
            chunk, label = self._get_from_h5(idx)

        if self.transform:
            chunk = self.transform(chunk)

        return chunk, label
