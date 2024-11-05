import os
import mne
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
_resample_freq = 250
_overlap = 0.5

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

    def build(self, outfile, overwrite=False, chunk_duration=_chunk_duration, resample_freq=_resample_freq, overlap=_overlap):
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
                    # preload = True to load the data into memory, otherwise it takes forever for a single cnt file
                    raw_data = mne.io.read_raw_cnt(cnt_file, data_format='int32', preload=True, verbose=False)
                    n_channels = raw_data.info["nchan"]
                    
                    s_freq = raw_data.info["sfreq"]
                    if resample_freq:
                        raw_data.resample(resample_freq, verbose=False)
                        s_freq = resample_freq
                    raw_data.filter(1, 50, verbose=False)
                    raw_data = raw_data.get_data()[:n_channels]
                except Exception as e:
                    print(f"Error processing {cnt_file}: {e}")
                    continue  # Skip to the next file

                session_info = session_labels[int(sid)]
                n_samples = int(chunk_duration * s_freq)

                for start_sec, end_sec, label in zip(session_info["start"], session_info["end"], session_info["labels"]):
                    start_idx = int(start_sec * s_freq)
                    end_idx = int(end_sec * s_freq)
                    emotion = _label_to_emotion[label]

                    overlap_samples = int(n_samples * overlap)

                    # iterate through chunks
                    for i in range(start_idx, end_idx, n_samples - overlap_samples):
                        chunk = raw_data[:, i:i+n_samples]
                        if chunk.shape[1] < n_samples: # ignore the last chunk if it's too short
                            continue
                        p_group = f.require_group(f'{pid}')
                        s_group = p_group.require_group(f'{sid}')
                        e_group = s_group.require_group(emotion)

                        chunk_id = f"{pid}_{sid}_{emotion}_{i}"
                        e_group.create_dataset(chunk_id, data=chunk, chunks=True, compression="gzip")

        print(f"Dataset with frequency {s_freq} Hz and chunk duration {chunk_duration} sec saved to {outfile_path}.")