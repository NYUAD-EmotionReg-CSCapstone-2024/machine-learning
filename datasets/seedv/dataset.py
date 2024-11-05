import os

import numpy as np
import h5py as h5

from torch.utils.data.dataset import Dataset

from .channel_mappings import _channel_mappings as channel_mappings

# CONSTANTS
_emotion_to_label = {
    "happy": 0,
    "sad": 1,
    "fear": 2,
    "neutral": 3,
    "angry": 4
}

# Exhastive list of participants, sessions, emotions, and channels
_participants = list(range(1, 16))
_sessions = list(range(1, 4))
_emotions = list(_emotion_to_label.keys())
_channels = list(channel_mappings.keys())

class SeedVDataset(Dataset):
    def __init__(self, root, h5file, transform=None, participants=_participants, sessions=_sessions, emotions=_emotions, channels=_channels):
        '''
        root: str
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
        '''
        self.root = root
        self.h5file = h5.File(os.path.join(root, h5file), "r")
        self.transform = transform
        self.data_ids = []

        self.participants = participants
        self.sessions = sessions
        self.emotions = emotions
        self.channels = channels

        self.validate_params()
        self.collect_data_ids()
        
        self.channel_ids = [channel_mappings[channel]["index"] for channel in self.channels]

    def validate_params(self):
        '''
        Validate the parameters passed to the dataset. All provided participants, sessions and emotions must be present in the dataset.
        Validate the name of channels.
        '''
        # for pid in self.participants:
        #     if str(pid) not in self.h5file:
        #         raise ValueError(f"Participant {pid} not found in the dataset.")
        #     for sid in self.sessions:
        #         if str(sid) not in self.h5file[str(pid)]:
        #             raise ValueError(f"Session {sid} not found for participant {pid}.")
        #         for emotion in self.emotions:
        #             if emotion not in self.h5file[str(pid)][str(sid)]:
        #                 raise ValueError(f"Emotion {emotion} not found for participant {pid} in session {sid}.")
        # for channel in self.channels:
        #     if channel not in channel_mappings:
        #         raise ValueError(f"Channel {channel} not found in the channel mappings.")

        for pid in self.participants:
            if str(pid) not in self.h5file:
               print(f"Participant {pid} not found in the dataset.")
               continue
            for sid in self.sessions:
                if str(sid) not in self.h5file[str(pid)]:
                    print(f"Session {sid} not found for participant {pid}.")
                    continue
                for emotion in self.emotions:
                    if emotion not in self.h5file[str(pid)][str(sid)]:
                        print(f"Emotion {emotion} not found for participant {pid} in session {sid}.")
                        continue
        for channel in self.channels:
            if channel not in channel_mappings:
                print(f"Channel {channel} not found in the channel mappings.")
                continue


    def collect_data_ids(self):
        for pid in self.participants:
            for sid in self.sessions:
                for emotion in self.emotions:
                    data_ids = list(self.h5file[str(pid)][str(sid)][emotion].keys())
                    self.data_ids.extend(data_ids)
        np.random.shuffle(self.data_ids)
        

    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        pid, sid, emotion, start_idx = data_id.split("_")
        start_idx = int(start_idx)
        chunk = self.h5file[pid][sid][emotion][data_id][()]
        chunk = chunk[self.channel_ids]
        if self.transform:
            chunk = self.transform(chunk)
        return chunk, _emotion_to_label[emotion]