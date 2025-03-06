import os
import pylsl
import yaml
import argparse
import numpy as np
import torch
import time
from collections import deque

# Import model architecture
from models import ATCNet  # Ensure this matches your actual model import path

def load_config(config_name):
    """Load the interface configuration file."""
    config_path = f"./config/interface/{config_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    """Main function for real-time EEG classification."""
    
    # Load configuration
    config = load_config(args.config)

    # Extract Parameters
    MODEL_FILEPATH = config["model_filepath"]
    WINDOW_SIZE = config["window_size"]
    SAMPLE_FREQUENCY = config["sample_frequency"]
    POLLING_FREQUENCY = config["polling_frequency"]

    # Calculate BUFFER_SIZE
    BUFFER_SIZE = WINDOW_SIZE * SAMPLE_FREQUENCY

    # Load Trained ATCNet Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ATCNet().to(device)
    model.load_state_dict(torch.load(MODEL_FILEPATH, map_location=device))
    model.eval()  

    # Connect to EEG Headset (LSL)
    print("Looking for EEG stream...")
    streams = pylsl.resolve_stream('type', 'EEG')
    inlet = pylsl.StreamInlet(streams[0])

    # Initialize EEG Buffer (Sliding Window)
    eeg_buffer = deque(maxlen=BUFFER_SIZE)
    print(f"EEG Stream connected. Awaiting data... (Window Size: {WINDOW_SIZE}s, Sample Rate: {SAMPLE_FREQUENCY}Hz)")

    while True:
        try:
            # Receive Real-Time EEG Data
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                eeg_buffer.append(sample)

                # Ensure Buffer is Full Before Inference
                if len(eeg_buffer) == BUFFER_SIZE:
                    # Convert EEG buffer to NumPy array
                    eeg_data = np.array(eeg_buffer).T  # Shape: [Channels x Timepoints]

                    # Preprocess Data
                    eeg_data = (eeg_data - np.mean(eeg_data)) / np.std(eeg_data)  # Normalize

                    # Convert to Torch Tensor and Add Batch Dimension
                    processed_data = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(device)

                    # Run Model Inference
                    with torch.no_grad():
                        prediction = model(processed_data)
                        predicted_class = torch.argmax(prediction, dim=1).item()

                    # Output the Classification Result
                    print(f"Predicted Emotion Class: {predicted_class}")

            time.sleep(POLLING_FREQUENCY)  

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Interface for Real-Time Classification")
    parser.add_argument("--config", type=str, required=True, help="Configuration file name (without .yaml)")
    args = parser.parse_args()
    main(args)
