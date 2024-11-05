import os
import mne

import numpy as np
import h5py as h5

from tqdm import tqdm

from .session_labels import _original_session_labels as session_labels

# CONSTANTS
_emotion_to_label = {
    "disgust": 0,
    "fear": 1,
    "sad": 2,
    "neutral": 3,
    "happy": 4
}

_label_to_emotion = {
    v:k 
    for k, v in _emotion_to_label.items()
}

# DEFAULTS
_chunk_duration = 1
_resample_freq = 200
_overlap = 0.5

_channels_to_drop = ['M1', 'M2', 'VEO', 'HEO']

# (n_channels, n_samples) -> (n_samples, n_channels, n_channels)
def _spatial_transform_62(eeg_data):
    if eeg_data.shape[0] != 62:
        raise ValueError("Reading must contain 62 channels.")
    # create 3D matrix
    n_samples = eeg_data.shape[1]
    spatial_data = np.zeros((n_samples, 9, 9))
    channel_mapping = [
        (0, 2), (0, 4), (0, 6),  # Fp1, Fpz, Fp2
        (1, 3), (1, 5),         # AF3, AF4
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),  # F7, F5, F3, F1, Fz, F2, F4, F6, F8
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),  # FT7, FC5, FC3, FC1, FCz, FC2, FC4, FC6, FT8
        (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),  # T7, C5, C3, C1, Cz, C2, C4, C6, T8
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),  # TP7, CP5, CP3, CP1, CPz, CP2, CP4, CP6, TP8
        (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8),  # P7, P5, P3, P1, Pz, P2, P4, P6, P8
        (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6),  # PO7, PO5, PO3, POz, PO4, PO6
        (8, 2), (8, 3), (8, 4), (8, 5), (8, 6)  # CB1, O1, Oz, O2, CB2
    ]
    for idx, (x, y) in enumerate(channel_mapping):
        spatial_data[:, x, y] = eeg_data[idx, :]
    return spatial_data

class SeedVBuilder:
    '''
    Builder class for the SEED-V dataset. The dataset is built from the raw EEG data and the session labels.
    The dataset is saved as an HDF5 file with the following structure:
    - pid
        - sid
            - emotion
                - chunk_id: numpy array of shape (n_channels, n_samples)
    root_dir: str
        Path to the directory containing the SEED-V dataset
    '''
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        raw_files = os.listdir(os.path.join(root_dir, "EEG_raw"))
        self.cnt_files = [
            os.path.join(root_dir, "EEG_raw", file)
            for file in raw_files 
            if file.endswith(".cnt")
        ]

    def build(
            self, 
            outfile, 
            overwrite=False, 
            chunk_duration=_chunk_duration, 
            resample_freq=_resample_freq, 
            overlap=_overlap,
            transform_spatial=False,
        ):
        '''
        Build the SEED-V dataset and save it to an HDF5 file.
        outfile: str
            HDF5 output file name
        chunk_duration: int
            Duration of each chunk in seconds
        resample_freq: int
            Frequency to resample the EEG data to
        overlap: int
            Overlap between consecutive chunks in percent (0-1)
        '''
        outfile_path = os.path.join(self.root_dir, outfile)
        
        file_exists = os.path.exists(outfile_path)
        if file_exists:
            open_mode = "a" if not overwrite else "w"
        else:
            open_mode = "w"

        with h5.File(outfile_path, open_mode) as f:
            for cnt_file in tqdm(self.cnt_files, desc="Processing files"):
                file_name = os.path.basename(cnt_file)
                pid, sid, _ = file_name.replace(".cnt", "").split("_")

                if file_exists and pid in f and sid in f[pid]:
                    print(f"Skipping {cnt_file} as it already exists in the dataset.")
                    continue
                
                try:
                    # preload = True to load the data into memory, otherwise it takes forever
                    raw_data = mne.io.read_raw_cnt(cnt_file, data_format='int32', preload=True, verbose=False)
                    raw_data.drop_channels(_channels_to_drop) # drop unused channels
                    
                    s_freq = raw_data.info["sfreq"]
                    if resample_freq:
                        raw_data.resample(resample_freq, verbose=False)
                        s_freq = resample_freq # update s_freq to new freq
                    raw_data.filter(1, 50, verbose=False)
                    raw_data = raw_data.get_data()
                except Exception as e:
                    print(f"Error processing {cnt_file}: {e}")
                    continue  # Skip to the next file

                session_info = session_labels[int(sid)]
                n_samples = int(chunk_duration * s_freq)

                # data is in (n_channels, n_samples) format
                for start_sec, end_sec, label in zip(session_info["start"], session_info["end"], session_info["labels"]):
                    start_idx = int(start_sec * s_freq)
                    end_idx = int(end_sec * s_freq)

                    overlap_samples = int(n_samples * overlap)

                    # iterate through chunks
                    for i in range(start_idx, end_idx, n_samples - overlap_samples):
                        chunk = raw_data[:, i:i+n_samples]
                        breakpoint()
                        if chunk.shape[1] < n_samples: # ignore the last chunk if it's too short
                            continue
                        p_group = f.require_group(str(pid))
                        s_group = p_group.require_group(str(sid))
                        e_group = s_group.require_group(str(label))

                        chunk_id = f"{pid}_{sid}_{label}_{i}"
                        if transform_spatial:
                            # (n_samples, n_channels(62)) -> (n_samples, 9, 9)
                            chunk = _spatial_transform_62(chunk)
                        e_group.create_dataset(chunk_id, data=chunk, chunks=True, compression="gzip")

        print(f"Dataset with frequency {s_freq} Hz and chunk duration {chunk_duration} sec saved to {outfile_path}.")