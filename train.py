import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import DatasetFactory, SplitterFactory
from models import ModelFactory, OptimizerFactory, SchedulerFactory

from trainers import Trainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

loss_fn = nn.CrossEntropyLoss()

def main(args):
    config_path = os.path.join("./config/experiments", f"{args.config}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"])

    dataset = DatasetFactory.create(
        config["dataset"]["name"],
        **config["dataset"]["params"],
        load=args.load
    )
    splitter = SplitterFactory.create(
        config["splitter"]["name"], 
        dataset=dataset,
        **config["splitter"]["params"]
    )

    train_set = splitter.trainset
    test_set = splitter.testset

    batch_size = config["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = ModelFactory.create(
        config["model"]["name"], 
        **config["model"]["params"]
    ).to(device)

    optimizer = OptimizerFactory.create(
        config["optimizer"]["name"], 
        model.parameters(),
        **config["optimizer"]["params"]
    )

    # Create scheduler
    scheduler = None
    if "scheduler" in config:
        scheduler = SchedulerFactory.create(
            config["scheduler"]["name"],
            optimizer,
            **config["scheduler"]["params"]
        )

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        exp_dir=os.path.join(config["exp_dir"], f"{args.config}"),
    )

    # Set scheduler if it exists
    if scheduler is not None:
        trainer.set_scheduler(scheduler)

    trainer.train(
        epochs=config["epochs"], 
        eval_every=config["eval_every"], 
        patience=config["patience"],
        resume=args.resume
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the SEED-V dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (without .yaml extension)")
    parser.add_argument("--load", action="store_true", help="Load the dataset in memory for faster training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    args = parser.parse_args()
    main(args)