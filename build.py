import os
import yaml
import argparse
from datasets.seedv.builder import SeedVBuilder

# Mandatory and optional parameters
MANDATORY_PARAMS = ["root_dir", "dataset", "outfile"]
DEFAULT_PARAMS = {
    "chunk_duration": 1,
    "overlap": 0,
}
OPTIONAL_PARAMS = ["notch_freq", "bandpass_freqs", "resample_freq", "normalize"]

# Combine all params (mandatory + optional)
ALL_PARAMS = MANDATORY_PARAMS + list(DEFAULT_PARAMS.keys()) + OPTIONAL_PARAMS

def load_config(config_file):
    """Load and return the configuration from a YAML file."""
    config_path = os.path.join("./configs/datasets", f"{config_file}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def validate_params(config):
    """Ensure the config contains only valid and mandatory parameters."""
    unknown_params = [param for param in config if param not in ALL_PARAMS]
    if unknown_params:
        raise ValueError(f"Invalid parameter(s): {', '.join(unknown_params)}")
    
    missing_params = [param for param in MANDATORY_PARAMS if param not in config]
    if missing_params:
        raise ValueError(f"Missing mandatory parameter(s): {', '.join(missing_params)}")
    
    for param, value in DEFAULT_PARAMS.items():
        if param not in config:
            config[param] = value

def validate_values(config):
    """Validate the values of the configuration parameters."""
    if config["dataset"] not in ["seedv"]:
        raise ValueError(f"Invalid dataset: {config['dataset']}")
    if not (0 <= config["overlap"] <= 1):
        raise ValueError("Overlap must be between 0 and 1")
    if "notch_freq" in config and config["notch_freq"] <= 0:
        raise ValueError("Notch frequency must be positive")
    if "bandpass_freqs" in config:
        if len(config["bandpass_freqs"]) != 2 or config["bandpass_freqs"][0] >= config["bandpass_freqs"][1]:
            raise ValueError("Bandpass frequencies must be two positive values, low < high")
    if "resample_freq" in config and config["resample_freq"] <= 0:
        raise ValueError("Resample frequency must be positive")

def create_preprocessors(config):
    """Create the list of preprocessors based on the config."""
    preprocessors = []
    if "notch_freq" in config:
        preprocessors.append(("notch_filter", {"notch_freq": config["notch_freq"]}))
    if "bandpass_freqs" in config:
        preprocessors.append(("bandpass_filter", {"low_freq": config["bandpass_freqs"][0], "high_freq": config["bandpass_freqs"][1]}))
    if "resample_freq" in config:
        preprocessors.append(("resample", {"target_freq": config["resample_freq"]}))
    if "normalize" in config and config["normalize"]:
        preprocessors.append(("normalize", {}))
    return preprocessors

def build_dataset(config, overwrite):
    """Build the SEED-V dataset using the provided config."""
    builder = SeedVBuilder(root_dir=config["root_dir"])
    builder.build(
        outfile=f"{config['outfile']}.h5", # h5file
        overwrite=overwrite,
        chunk_duration=config["chunk_duration"],
        overlap=config["overlap"],
        preprocessors=create_preprocessors(config)
    )

def main(config_file, overwrite):
    """Main function to load the config, validate and build the dataset."""
    config = load_config(config_file)
    validate_params(config)
    validate_values(config)
    
    if config["dataset"] == "seedv":
        build_dataset(config, overwrite)
    else:
        raise ValueError(f"Dataset '{config['dataset']}' is not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the SEED-V dataset")
    parser.add_argument("--config", required=True, help="Path to the configuration YAML file (without extension)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing dataset")
    
    args = parser.parse_args()
    main(args.config, args.overwrite)