import mne
import numpy as np

class EEGPreprocessor:
    def __init__(self):
        self.preprocess_methods = {
            'notch_filter': self.apply_notch_filter,
            'eog_removal': self.remove_eog_artifacts,
            'bandpass_filter': self.bandpass_filter,
            'resample': self.resample,
            'normalize': self.normalize
        }

    def apply_notch_filter(_, raw_data, notch_freq=50):
        """
        Apply a notch filter to remove specified frequency noise (e.g., 50 Hz power line noise).
        
        Parameters:
        - notch_freq: int
            Frequency to notch filter (default: 50 Hz).
        """
        raw_data.notch_filter(freqs=notch_freq, picks="eeg", verbose=False)
        return raw_data # for clarity, chaining, and consistency

    def remove_eog_artifacts(_, raw_data, eog_channels=['EOG', 'FPZ']):
        """
        Use ICA to remove EOG artifacts from EEG data.
        
        Parameters:
        - eog_channels: list of str
            List of EOG channel names (default: ['EOG', 'FPZ']).
        """
        ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter='auto', verbose=False)
        ica.fit(raw_data)
        eog_inds, _ = ica.find_bads_eog(raw_data, ch_name=eog_channels)
        ica.exclude = eog_inds
        raw_data = ica.apply(raw_data, verbose=False)
        return raw_data

    def bandpass_filter(_, raw_data, low_freq=1.0, high_freq=50.0):
        """
        Apply a bandpass filter to keep frequencies within a specified range.
        
        Parameters:
        - low_freq: float
            Lower frequency bound (default: 1.0 Hz).
        - high_freq: float
            Upper frequency bound (default: 50.0 Hz).
        """
        raw_data.filter(l_freq=low_freq, h_freq=high_freq, picks="eeg", verbose=False)
        return raw_data

    def resample(_, raw_data, target_freq=200):
        """
        Resample and normalize the EEG data to a target frequency.
        
        Parameters:
        - target_freq: int
            Target frequency for resampling (default: 200 Hz).
        """
        raw_data.resample(target_freq, verbose=False)
        return raw_data

    def normalize(self, raw_data):
        """
        Normalize the EEG data per participant.
        """
        # Apply function to each channel individually
        raw_data.apply_function(
            lambda x: (x - np.mean(x)) / np.std(x),
            picks="eeg",
            verbose=False
        )
        return raw_data


        
    def preprocess(self, raw_data, steps):
        """
        Apply preprocessing steps in the specified order.
        
        Parameters:
        - steps: list of tuples
            List of tuples where each tuple contains the method name as a string and a dictionary of arguments.
            Example: [('notch_filter', {'notch_freq': 50}), ('bandpass_filter', {'low_freq': 1, 'high_freq': 50})]
        """
        if not steps:
            return raw_data
        for step, args in steps:
            if step not in self.preprocess_methods:
                raise ValueError(f"Preprocessing method '{step}' not found.")
            raw_data = self.preprocess_methods[step](raw_data, **args)
        return raw_data