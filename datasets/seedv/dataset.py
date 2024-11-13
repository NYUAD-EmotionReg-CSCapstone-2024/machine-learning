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
    def __init__(self, root_dir, h5file, transform=None, participants=_participants, sessions=_sessions, emotions=_emotions, channels=_channels):
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
        self.transform = transform
        self.data_ids = []

        self.participants = participants
        self.sessions = sessions
        self.emotions = emotions
        self.channels = channels

        self._validate_params()
        self._collect_data_ids()
        
        self.channel_ids = [channel_mappings[channel]["index"] for channel in self.channels]

    def _validate_params(self):
        '''
        Validate the parameters passed to the dataset. All provided participants, sessions and emotions must be present in the dataset.
        Validate the name of channels.
        '''
        for pid in self.participants:
            if str(pid) not in self.h5file:
                raise ValueError(f"Participant {pid} not found in the dataset.")
            for sid in self.sessions:
                # escape participant 7 session 1 (not working yet, need to fix)
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


    def _collect_data_ids(self):
        for pid in self.participants:
            for sid in self.sessions:
                # escape participant 7 session 1 (not working yet, need to fix)
                if str(pid) == "7" and str(sid) == "1":
                    continue
                for emotion in self.emotions:
                    data_ids = list(self.h5file[str(pid)][str(sid)][str(emotion)].keys())
                    self.data_ids.extend(data_ids)
        np.random.shuffle(self.data_ids)


    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        pid, sid, emotion, _ = data_id.split("_")

        chunk = self.h5file[pid][sid][emotion][data_id][()]
        chunk = chunk[self.channel_ids]
        
        chunk = torch.tensor(chunk, dtype=torch.float32)
        label = torch.tensor(int(emotion), dtype=torch.long)

        if self.transform:
            chunk = self.transform(chunk)

        return chunk, label