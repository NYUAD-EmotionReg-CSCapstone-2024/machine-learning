import os
import mne
import warnings

import h5py as h5

from tqdm import tqdm

from ..EEGPreprocessor import EEGPreprocessor
from .session_labels import _original_session_labels as session_labels

# Ignore cnt file warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

_channels_to_drop = ['M1', 'M2', 'VEO', 'HEO']

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
        self.cnt_files = [
            os.path.join(root_dir, "EEG_raw", file)
            for file in raw_files 
            if file.endswith(".cnt")
        ]

    def build(
            self, 
            outfile, 
            overwrite,
            chunk_duration, 
            overlap,
            preprocessors
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
                    raw_data = eeg_preprocessor.preprocess(raw_data, preprocessors) # apply diff filters
                    s_freq = raw_data.info["sfreq"]
                    raw_data = raw_data.get_data() # (n_samples, n_channels) ndarray
                except Exception as e:
                    print(f"Error processing {cnt_file}: {e}")
                    continue  # Skip to the next file

                session_info = session_labels[int(sid)] # ground truth labels for the sessions
                n_samples = int(chunk_duration * s_freq)
                for start_sec, end_sec, label in zip(session_info["start"], session_info["end"], session_info["labels"]):
                    start_idx = int(start_sec * s_freq)
                    end_idx = int(end_sec * s_freq)

                    overlap_samples = int(n_samples * overlap)

                    # iterate through chunks
                    for i in range(start_idx, end_idx, n_samples - overlap_samples):
                        chunk = raw_data[:, i:i+n_samples]
                        if chunk.shape[1] < n_samples: # ignore the last chunk if it's too short
                            continue
                        p_group = f.require_group(str(pid))
                        s_group = p_group.require_group(str(sid))
                        e_group = s_group.require_group(str(label))

                        chunk_id = f"{pid}_{sid}_{label}_{i}"
                        e_group.create_dataset(chunk_id, data=chunk, chunks=True, compression="gzip")

        print(f"Dataset with frequency {s_freq} Hz and chunk duration {chunk_duration} sec saved to {outfile_path}.")