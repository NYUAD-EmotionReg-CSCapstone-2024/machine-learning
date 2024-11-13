import os
import pickle
import yaml
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.seedv.dataset import SeedVDataset
from datasets.splitters.RandomSplitter import RandomSplit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
    models = {
        "conv_transformer": "models.conv_transformer.BaseModel",
        "ertnet": "models.ERTNet.ERTNet",
        "atcnet": "models.atcnet.ATCNet"
    }
    if model_name in models:
        module_name, class_name = models[model_name].rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)()
    raise ValueError(f"Invalid model: {model_name}")

def get_loss_fn(loss_fn_name):
    loss_fns = {
        "cross_entropy": nn.CrossEntropyLoss,
        "mse": nn.MSELoss
    }
    if loss_fn_name in loss_fns:
        return loss_fns[loss_fn_name]()
    raise ValueError(f"Invalid loss function: {loss_fn_name}")


def get_optimizer(model, optimizer_name, lr=1e-3):
    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD
    }
    if optimizer_name in optimizers:
        return optimizers[optimizer_name](model.parameters(), lr=lr)
    raise ValueError(f"Invalid optimizer: {optimizer_name}")


class TrainLoop:
    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Setup model, loss function, and optimizer
        self.model = get_model(config["model"]).to(device)
        self.criterion = get_loss_fn(config["loss_fn"])
        self.optimizer = get_optimizer(self.model, config["optimizer"], lr=config["lr"])

        # Directory setup
        self.experiment_dir = os.path.join(config["exp_dir"], "exp_" + str(config["exp_num"]))
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        loss_values, avg_loss_values, acc_values = [], [], []

        for epoch in tqdm(range(self.config["epochs"]), desc="Epochs"):
            avg_loss = self._train_epoch()
            avg_loss_values.append(avg_loss)

            if (epoch + 1) % self.config["eval_every"] == 0:
                accuracy = self.evaluate()
                acc_values.append(accuracy)
                tqdm.write(f"Epoch: {epoch+1}, Accuracy: {accuracy:.4f}")

            if (epoch + 1) % self.config["save_every"] == 0:
                self._save_checkpoint(epoch)

        self._save_final_model()
        self._save_metrics(loss_values, avg_loss_values, acc_values)
        self._plot_metrics(loss_values, avg_loss_values, acc_values)
        self._dump_info()

        print("Training completed successfully")

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        with tqdm(total=len(self.train_loader), desc="Training Iterations", leave=False) as epoch_pbar:
            for idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                epoch_pbar.set_postfix({"loss": running_loss / (idx + 1)})
                epoch_pbar.update()
        return running_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def _save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        tqdm.write(f"Checkpoint saved to {checkpoint_path}")

    def _save_final_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "final_model.pth"))

    def _save_metrics(self, loss_values, avg_loss_values, acc_values):
        with open(os.path.join(self.experiment_dir, "metrics.pkl"), "wb") as f:
            pickle.dump({"loss": loss_values, "avg_loss": avg_loss_values, "accuracy": acc_values}, f)

    def _plot_metrics(self, loss_values, avg_loss_values, acc_values):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(loss_values, label="Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(avg_loss_values, label="Average Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(acc_values, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "plot.png"))

    def _dump_info(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        with open(os.path.join(self.experiment_dir, "info.yaml"), "w") as f:
            yaml.safe_dump(self.config, f)
        with open(os.path.join(self.experiment_dir, "model_info.txt"), "w") as f:
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            f.write(f"Number of params: {num_params}\n")
            f.write(f"Model: {self.model}\n")

def main(args):
    config_path = os.path.join("./configs/experiments", f"exp_{args.config}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = SeedVDataset(
        root=config["root_dir"],
        h5file=f"{config['dataset']}.h5",
        participants=config["participants"],
    )

    # Split the dataset into train and test sets
    split = RandomSplit(dataset)
    train_set = split.trainset
    test_set = split.testset

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    # Initialize and run the training loop
    train_loop = TrainLoop(config, train_loader, test_loader)
    train_loop.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the SEED-V dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (without .yaml extension)")
    args = parser.parse_args()
    main(args)