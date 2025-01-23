import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import DatasetFactory, SplitterFactory
from models import ModelFactory, OptimizerFactory, SchedulerFactory  # Added SchedulerFactory
from trainers import Trainer

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting training script...")

loss_fn = nn.CrossEntropyLoss()

def main(args):
    config_path = os.path.join("./config/experiments", f"{args.config}.yaml")
    logging.info(f"Looking for config file at: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    logging.info("Loading configuration file...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully.")

    # Set device
    device = torch.device(config["device"])
    logging.info(f"Using device: {device}")

    # Create dataset
    logging.info("Creating dataset...")
    dataset = DatasetFactory.create(
        config["dataset"]["name"],
        **config["dataset"]["params"],
        load=args.load
    )
    logging.info("Dataset created successfully.")

    # Create splitter
    logging.info("Splitting dataset...")
    splitter = SplitterFactory.create(
        config["splitter"]["name"], 
        dataset=dataset,
        **config["splitter"]["params"]
    )
    logging.info("Dataset split successfully.")

    train_set = splitter.trainset
    test_set = splitter.testset

    # Create data loaders
    logging.info("Creating data loaders...")
    batch_size = config["batch_size"]

    # Define optimized DataLoader parameters
    num_workers = 16  # Adjust based on available CPU cores
    prefetch_factor = 2  # Number of batches prefetched by each worker

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Enables faster GPU data transfer
        drop_last=True,   # Ensures consistent batch size
        prefetch_factor=prefetch_factor,
        persistent_workers=True  # Reuse workers across epochs
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # Do not drop last batch for testing
        prefetch_factor=prefetch_factor,
        persistent_workers=True  # Reuse workers across epochs
    )

    logging.info("Data loaders created successfully.")


    # Create model
    logging.info("Creating model...")
    model = ModelFactory.create(
        config["model"]["name"], 
        **config["model"]["params"]
    ).to(device)
    logging.info("Model created successfully.")

    # Create optimizer
    logging.info("Creating optimizer...")
    optimizer = OptimizerFactory.create(
        config["optimizer"]["name"], 
        model.parameters(),
        **config["optimizer"]["params"]
    )
    logging.info("Optimizer created successfully.")

    # Create scheduler
    scheduler = None
    if "scheduler" in config:
        logging.info("Creating scheduler...")
        scheduler = SchedulerFactory.create(
            config["scheduler"]["name"],
            optimizer,
            **config["scheduler"]["params"]
        )
        logging.info("Scheduler created successfully.")

    # Create trainer
    logging.info("Creating trainer...")
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        exp_dir=os.path.join(config["exp_dir"], f"exp_{args.config}"),
    )
    logging.info("Trainer created successfully.")

    # Set scheduler if it exists
    if scheduler is not None:
        trainer.set_scheduler(scheduler)

    # Start training
    logging.info("Starting training...")
    trainer.train(
        epochs=config["epochs"], 
        eval_every=config["eval_every"], 
        patience=config["patience"],
        resume=args.resume
    )
    logging.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the SEED-V dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (without .yaml extension)")
    parser.add_argument("--load", action="store_true", help="Load the dataset in memory for faster training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    
    args = parser.parse_args()
    main(args)
