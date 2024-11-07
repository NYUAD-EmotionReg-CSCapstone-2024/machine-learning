import argparse

from datasets.seedv.builder import SeedVBuilder

SEEDV_ROOT = "/data/SEED-V"

# DEFAULTS
CHUNK_DURATION = 1
RESAMPLE_FREQ = 200
OVERLAP = 0.5


def main(args):
    if args.dataset == "seedv":
        builder = SeedVBuilder(root_dir=SEEDV_ROOT)
        builder.build(
            outfile=args.outfile,
            overwrite=args.overwrite,
            chunk_duration=args.chunk_duration,
            resample_freq=args.resample_freq,
            overlap=args.overlap,
        )
    else:
        raise ValueError(f"Not implemented yet: {args.dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the SEED-V dataset")

    parser.add_argument("--dataset", type=str, default="seedv", help="Dataset name")
    parser.add_argument("--outfile", type=str, default="seedv.h5", help="Output file name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists")
    parser.add_argument("--chunk-duration", type=int, default=CHUNK_DURATION, help="Duration of each chunk in seconds")
    parser.add_argument("--resample-freq", type=int, default=RESAMPLE_FREQ, help="Frequency to resample the EEG data to")
    parser.add_argument("--overlap", type=float, default=OVERLAP, help="Overlap between consecutive chunks in percent (0-1)")

    # parse and build
    args = parser.parse_args()
    main(args)