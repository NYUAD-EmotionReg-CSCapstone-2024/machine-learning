import os
import mne

class DEAPBuilder:
    def __init__(self, root_dir):
        self.root_idr = root_dir

        self.data_dir = os.path.join(root_dir, "data_original")
        raw_files = os.listdir(self.data_dir)
        self.bdf_files = [
            os.path.join(self.data_dir, file)
            for file in raw_files
            if file.endswith(".bdf")
        ]
    
    def build(self, outfile, overwrite, chunk_duration, overlap, preprocessors):
        for file in self.bdf_files:
            raw = mne.io.read_raw_bdf(file, preload=True)
            pass