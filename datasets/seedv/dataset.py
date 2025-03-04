import os
import torch

import numpy as np

import h5py as h5

from torch.utils.data.dataset import Dataset

from .channel_mappings import _channel_info as channel_mappings

_total_participants = 16
_total_sessions = 3
_total_emotions = 5

# Exhastive list of participants, sessions, emotions, and channels
_participants = [str(i) for i in range(1, _total_participants + 1)]
_sessions = [str(i) for i in range(1, _total_sessions + 1)]
_emotions = [str(emotion) for emotion in range(_total_emotions)]
_channels = [str(key) for key in channel_mappings.keys()]

class SeedVDataset(Dataset):
    def __init__(self, root_dir, h5file, device, transform=None, participants=_participants, sessions=_sessions, emotions=_emotions, channels=_channels, load=False):
        '''
        root_dir: str
            Path to the root directory containing the dataset
        h5file: str
            Path to the h5 file containing the dataset
        transform: callable
            Function to apply to each chunk
        participants: list
            List of participant IDs to include in the dataset
            Defaults to the full list of participants: [1, 2, 3, ..., 16]
        sessions: list
            List of session IDs to include in the dataset
            Defaults to the full list of sessions: [1, 2, 3]
        emotions: list
            List of emotions to include in the dataset
            Defaults to the full list of emotions: ["happy", "sad", "fear", "neutral", "angry"]
        channels: list
            List of channels to include in the dataset
        '''
        self.root_dir = root_dir
        self.h5file = h5.File(os.path.join(root_dir, h5file), "r")
        self.device = device

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
            self._load_in_memory()
        

    def _validate_params(self):
        '''
        Validate the parameters passed to the dataset. All provided participants, sessions and emotions must be present in the dataset.
        Validate the name of channels.
        '''
        for pid in self.participants:
            if str(pid) not in self.h5file:
                raise ValueError(f"Participant {pid} not found in the dataset.")
            for sid in self.sessions:
                if str(sid) not in self.h5file[str(pid)]:
                    raise ValueError(f"Session {sid} not found for participant {pid}.")
                for emotion in self.emotions:
                    if str(emotion) not in self.h5file[str(pid)][str(sid)]:
                        continue
        for channel in self.channels:
            if channel not in channel_mappings:
                raise ValueError(f"Channel {channel} not found in the channel mappings.")


    def _collect_data_ids(self):
        """Collect data_ids"""
        for pid in self.participants:
            for sid in self.sessions:
                for emotion in self.emotions:
                    if str(emotion) not in self.h5file[str(pid)][str(sid)]:
                        continue
                    data_ids = list(self.h5file[str(pid)][str(sid)][str(emotion)].keys())
                    self.data_ids.extend(data_ids)

        # Shuffle data IDs 
        np.random.shuffle(self.data_ids)

    def _load_in_memory(self):
        self.data = []
        self.labels = []
        for data_id in self.data_ids:
            pid, sid, emotion, _ = data_id.split("_")
            chunk = self.h5file[pid][sid][emotion][data_id][()]
            chunk = chunk[self.channel_ids]
            self.data.append(torch.tensor(chunk, dtype=torch.float32).to(self.device))
            self.labels.append(torch.tensor(int(emotion), dtype=torch.long).to(self.device))

    def __len__(self):
        return len(self.data_ids)
    
    def _get_from_memory(self, idx):
        chunk = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return chunk, label

    def _get_from_h5(self, idx):
        data_id = self.data_ids[idx]
        pid, sid, emotion, _ = data_id.split("_")
        chunk = self.h5file[pid][sid][emotion][data_id][()]
        chunk = chunk[self.channel_ids]
        chunk = torch.tensor(chunk, dtype=torch.float32)
        label = torch.tensor(int(emotion), dtype=torch.long)
        return chunk, label

    def __getitem__(self, idx):
        if self.load:
            chunk, label = self._get_from_memory(idx)
        else:
            chunk, label = self._get_from_h5(idx)

        if self.transform:
            chunk = self.transform(chunk)

        return chunk, label