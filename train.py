import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import DatasetFactory, SplitterFactory
from models import ModelFactory
from trainers import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()

def get_optimizer(model, optimizer_name, lr=1e-3):
    optimizers = {
        "adam": torch.optim.Adam,
    }
    if optimizer_name in optimizers:
        return optimizers[optimizer_name](model.parameters(), lr=lr)
    raise ValueError(f"Invalid optimizer: {optimizer_name}")


def main(args):
    config_path = os.path.join("./config/experiments", f"exp_{args.config}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = DatasetFactory.get_dataset(
        config["dataset"]["name"], 
        **config["dataset"]["params"]
    )
    splitter = SplitterFactory.get_splitter(
        config["splitter"]["name"], 
        dataset=dataset, 
        **config["splitter"]["params"]
    )

    train_set = splitter.trainset
    test_set = splitter.testset

    batch_size = config["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = ModelFactory.get_model(
        config["model"]["name"], 
        **config["model"]["params"]
    ).to(device)

    optimizer = get_optimizer(
        model, 
        config["optimizer"]["name"], 
        lr=config["optimizer"]["lr"]
    )

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        patience=config["patience"],
        exp_dir=os.path.join(config["root_dir"], config["exp_dir"], f"exp_{config['exp_num']}"),
    )

    trainer.train(config["epochs"], config["eval_every"], resume=args.resume)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the SEED-V dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (without .yaml extension)")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    args = parser.parse_args()
    main(args)