import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import DatasetFactory, SplitterFactory
from models import ModelFactory, OptimizerFactory, SchedulerFactory
from multiprocessing import Process, Manager
from trainers import Trainer
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

loss_fn = nn.CrossEntropyLoss()

def create_trainer(config, splitter, device, fold_idx=None):
    """Creates and returns a trainer for a given dataset split and configuration."""
    if fold_idx is not None:
        splitter.set_fold(fold_idx)

    train_loader = DataLoader(splitter.trainset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(splitter.testset, batch_size=config["batch_size"], shuffle=False)

    model = ModelFactory.create(
        config["model"]["name"], 
        **config["model"]["params"]
    ).to(device)

    optimizer = OptimizerFactory.create(
        config["optimizer"]["name"], 
        model.parameters(), 
        **config["optimizer"]["params"]
    )

    scheduler = SchedulerFactory.create(
        config["scheduler"]["name"], 
        optimizer, 
        **config["scheduler"]["params"]
    ) if "scheduler" in config else None

    exp_dir = os.path.join(config["exp_dir"], f"{args.config}_fold{fold_idx}" if fold_idx is not None else args.config)

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        exp_dir=exp_dir,
    )

    if scheduler:
        trainer.set_scheduler(scheduler)

    return trainer

def train_fold(fold_idx, config, splitter, device, results, args):
    """Function to train a single fold."""
    print(f"\nStarting Fold {fold_idx + 1}/{config['splitter']['params']['k']}")
    trainer = create_trainer(config, splitter, device, fold_idx)
    
    trainer.train(
        epochs=config["epochs"], 
        eval_every=config["eval_every"], 
        patience=config["patience"], 
        resume=args.resume
    )
    
    # Store metrics for this fold
    results[fold_idx] = trainer.metrics

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

    # Handle K-Fold Cross-Validation
    if config["splitter"]["name"] == "kfold":
        num_folds = config["splitter"]["params"]["k"]
        
        # Shared manager for storing results from each fold
        with Manager() as manager:
            results = manager.dict()
            processes = []

            # Parallel training for each fold idx
            for fold_idx in range(num_folds):
                p = Process(target=train_fold, args=(fold_idx, config, splitter, device, results, args))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # Aggregate results
            fold_metrics = [results[fold_idx] for fold_idx in range(num_folds)]
            avg_val_loss = sum([m["best_val_loss"] for m in fold_metrics]) / num_folds
            print(f"\nK-Fold Cross-Validation Completed\nAverage Validation Loss: {avg_val_loss:.4f}")
    
   # Other splitters such as Random and LNSO
    else:
        trainer = create_trainer(config, splitter, device)
        
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
