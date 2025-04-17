import os
import mne
import warnings

import h5py as h5
import pandas as pd
from tqdm import tqdm

from ..EEGPreprocessor import EEGPreprocessor
from .session_labels import _original_session_labels as session_labels

# Ignore cnt file warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

_channels_to_drop = ['M1', 'M2', 'VEO', 'HEO']
_scores_file_path = os.path.join(os.path.dirname(__file__), "scores.csv")

# Mapping 5 emotions to 3 categories
# Left: 0 - disgust, 1 - fear, 2 - sad, 3 - neutral, 4 - happy
# Right: 0 - negative, 1 - neutral, 2 - positive
label_map = {
    "0": "0",
    "1": "0",
    "2": "0",
    "3": "1",
    "4": "2"
}

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
    preprocessors: list
        List of preprocessors to apply to the raw EEG data
    '''
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        raw_files = os.listdir(os.path.join(root_dir, "EEG_raw"))
        self.eeg_files = [
            os.path.join(root_dir, "EEG_raw", file)
            for file in raw_files 
            if file.endswith(".cnt") or file.endswith(".edf")
        ]

        if not os.path.exists(_scores_file_path):
            raise FileNotFoundError(f"Scores file not found at {_scores_file_path}")
        try:
            self.scores_df = pd.read_csv(_scores_file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading scores file: {e}")
        
    def build(
            self, 
            outfile, 
            overwrite,
            chunk_duration,
            overlap, 
            preprocessors,
        ):
        '''
        Build the SEED-V dataset and save it to an HDF5 file.
        outfile: str
            HDF5 output file name
        overwrite: bool
            Whether to overwrite the existing file
       chunk_duration: int
            Duration of each chunk in seconds
        resample_freq: int
            Frequency to resample the EEG data to
        overlap: int
            Overlap between consecutive chunks in percent (0-1)
        preprocessors: list
            List of preprocessors to apply to the raw EEG data
            notch, bandpass, resample, eog_removal, normalize
        '''
        outfile_path = os.path.join(self.root_dir, outfile)
        
        file_exists = os.path.exists(outfile_path)
        open_mode = "w"
        if file_exists and not overwrite:
            open_mode = "a"

        eeg_preprocessor = EEGPreprocessor()
        s_freq = None

        with h5.File(outfile_path, open_mode) as f:
            for eeg_file in tqdm(self.eeg_files, desc="Processing files"):
                file_name = os.path.basename(eeg_file)

                # File 7_1 is in a different format due to corruption
                if file_name.endswith(".cnt"):
                    pid, sid, _ = file_name.replace(".cnt", "").split("_")
                elif file_name.endswith(".edf"): 
                    pid, sid, _ = file_name.replace(".edf", "").split("_")

                if file_exists and pid in f and sid in f[pid]:
                    print(f"Skipping {eeg_file} as it already exists in the dataset.")
                    continue
                
                try:
                    # preload = True to load the data into memory, otherwise it takes forever
                    if(eeg_file.endswith(".cnt")):
                        raw_data = mne.io.read_raw_cnt(eeg_file, data_format='int32', preload=True, verbose=False)
                    elif(eeg_file.endswith(".edf")):
                        raw_data = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
                    raw_data.drop_channels(_channels_to_drop) # drop unused channels
                    raw_data = eeg_preprocessor.preprocess(raw_data, preprocessors) # apply diff filters
                    s_freq = raw_data.info["sfreq"]
                    raw_data = raw_data.get_data() # (n_samples, n_channels) ndarray
                except Exception as e:
                    print(f"Error processing {eeg_file}: {e}")
                    continue  # Skip to the next file

                session_info = session_labels[int(sid)] # Ground truth labels for the sessions
                n_samples = int(chunk_duration * s_freq)
                for mid, (start_sec, end_sec, label) in enumerate(zip(session_info["start"], session_info["end"], session_info["labels"])):
                    # Check if this movie is valid for processing based on scores.csv
                    score_row = self.scores_df[
                        (self.scores_df["pid"] == int(pid)) & 
                        (self.scores_df["sid"] == int(sid)) & 
                        (self.scores_df["mid"] == mid + 1)
                    ]
                    if score_row.empty or score_row.iloc[0]["binary"] == 0:
                        continue  # Skip this movie if binary flag is 0

                    # Map 5-class to 3-class - Negative, Neutral, Positive
                    mapped_label = label_map.get(str(label))
                    if not mapped_label:
                        continue
                    
                    start_idx = int(start_sec * s_freq)
                    end_idx = int(end_sec * s_freq)
                    overlap_samples = int(n_samples * overlap)

                    # Iterate through chunks 
                    for i in range(start_idx, end_idx, n_samples - overlap_samples):
                        chunk = raw_data[:, i:i+n_samples]
                        if chunk.shape[1] < n_samples: # Ignore the last chunk if it's too short
                            continue

                        p_group = f.require_group(str(pid))
                        s_group = p_group.require_group(str(sid))
                        e_group = s_group.require_group(str(mapped_label))

                        chunk_id = f"{pid}_{sid}_{label}_{mapped_label}_{i}"
                        e_group.create_dataset(chunk_id, data=chunk, chunks=True, compression="gzip")

        print(f"Dataset with frequency {s_freq} Hz and chunk duration {chunk_duration} sec saved to {outfile_path}.")