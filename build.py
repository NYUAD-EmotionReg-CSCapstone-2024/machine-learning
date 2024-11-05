import argparse

from datasets.seedv.builder import SeedVBuilder

SEEDV_ROOT = "/data/SEED-V"

def main(args):
    if args.dataset == "seedv":
        builder = SeedVBuilder(root_dir=SEEDV_ROOT)
        builder.build(
            outfile=args.outfile,
            chunk_duration=args.chunk_duration,
            resample_freq=args.resample_freq,
            overlap=args.overlap
        )
    else:
        raise ValueError(f"Not implemented yet: {args.dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the SEED-V dataset")

    parser.add_argument("--dataset", type=str, default="seedv", help="Dataset name")
    parser.add_argument("--outfile", type=str, default="seedv.h5", help="Output file name")
    parser.add_argument("--chunk_duration", type=int, default=1, help="Duration of each chunk in seconds")
    parser.add_argument("--resample_freq", type=int, default=250, help="Frequency to resample the EEG data to")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap between consecutive chunks in percent (0-1)")
    
    # parse and build
    args = parser.parse_args()
    main(args)