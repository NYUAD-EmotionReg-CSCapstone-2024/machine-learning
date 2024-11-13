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

def get_optimizer(model, optimizer_name, **kwargs):
    optimizers = {
        "adam": {
            "optimizer": torch.optim.Adam,
            "mandatory_params": ["lr"]
        }
    }
    if optimizer_name in optimizers:
        config = optimizers[optimizer_name]
        for param in config["mandatory_params"]:
            if param not in kwargs:
                raise ValueError(f"Missing parameter: {param}")
        return config["optimizer"](model.parameters(), **kwargs)
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
        **config["optimizer"]["params"]
    )

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        exp_dir=os.path.join(config["exp_dir"], f"exp_{config['exp_num']}"),
    )

    trainer.train(
        epochs=config["epochs"], 
        eval_every=config["eval_every"], 
        patience=config["patience"],
        resume=args.resume
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the SEED-V dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (without .yaml extension)")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    args = parser.parse_args()
    main(args)