import os
import shutil
import logging
import torch

import matplotlib.pyplot as plt

from abc import ABC
from tqdm import tqdm

class Trainer(ABC):
    def __init__(self, train_loader, val_loader, model, loss_fn, optimizer, patience, exp_dir):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.patience = patience
        self.exp_dir = exp_dir

        self._setup_directories(exp_dir)
        self.early_stopping_counter = 0
        self.metrics = {
            "best_val_loss": float("inf"),
            "train_loss": [],
            "val_loss": [],
            "acc": []
        }

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_logger(self, exp_dir, mode="w"):
        """Set up the logger to log training information."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        log_file = os.path.join(exp_dir, "train.log")
        file_handler = logging.FileHandler(log_file, mode=mode)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _setup_directories(self, exp_dir):
        """Create necessary directories for saving checkpoints and logs."""
        self.checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        self.latest_checkpoint = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _cleanup_dir(self, exp_dir):
        """Clean up directories before starting fresh training."""
        for file in os.listdir(exp_dir):
            file_path = os.path.join(exp_dir, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        self._setup_directories(exp_dir)

    def _log_model_info(self):
        """Log model, loss function, and optimizer info."""
        self.logger.info("Model:\n{}".format(self.model))
        self.logger.info("Loss Function:\n{}".format(self.loss_fn))
        self.logger.info("Optimizer:\n{}".format(self.optimizer))

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics = checkpoint["metrics"]
        epoch = checkpoint["epoch"]
        self.logger.info(f"\n\nResuming from epoch {epoch + 1}")
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return epoch

    def _save_checkpoint(self, epoch, checkpoint_path, latest=False):
        """Save checkpoint at the end of each epoch."""
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if latest:
            checkpoint_data["epoch"] = epoch
            checkpoint_data["metrics"] = self.metrics
        torch.save(checkpoint_data, checkpoint_path)

    def _evaluate(self):
        """Evaluate model performance on the validation set."""
        self.model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc="Validation Iterations", leave=False) as pbar:
                for data, labels in self.val_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = self.model(data)
                    val_loss += self.loss_fn(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    pbar.update()
        acc = correct / total
        val_loss = val_loss / len(self.val_loader)
        return acc, val_loss

    def _train_epoch(self):
        """Train the model for one epoch."""
        running_loss = 0.0
        with tqdm(total=len(self.train_loader), desc="Training Iterations", leave=False) as pbar:
            for idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"loss": running_loss / (idx + 1)})
                pbar.update()
        return running_loss / len(self.train_loader)

    def _resume_training(self):
        """Resume training from the last saved checkpoint."""
        # check if latest checkpoint exists
        latest_checkpoint = os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        if not os.path.exists(latest_checkpoint):
            self._setup_logger(self.exp_dir, mode="w")
            self.logger.info("No checkpoint found to resume training")
            self.logger.info("Starting fresh training")
            return 0

        self._setup_logger(self.exp_dir, mode="a")
        return self._load_checkpoint(latest_checkpoint)

    def _prepare_for_training(self, resume):
        """Prepare either for fresh training or resume from a checkpoint."""
        start_epoch = 0
        if resume:
            start_epoch = self._resume_training()
            if start_epoch == 0:
                self._cleanup_dir(self.exp_dir)
                self._setup_logger(self.exp_dir, mode="w")
                self._log_model_info()
        else:
            self._cleanup_dir(self.exp_dir)
            self._setup_logger(self.exp_dir, mode="w")
            self._log_model_info()
        return start_epoch
    
    def _plot_metrics(self, eval_every):
        # plot train_loss, validation_loss, and val_acc
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        eval_epochs = list(range(eval_every, len(self.metrics["val_loss"]) * eval_every + 1, eval_every))

        # Plot training loss (x-axis: epochs, y-axis: train loss)
        ax[0].plot(self.metrics["train_loss"], label="Train Loss")
        ax[0].set_title("Training Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # Plot validation loss (x-axis: evaluation epochs, y-axis: val loss)
        ax[1].plot(eval_epochs, self.metrics["val_loss"], label="Validation Loss", color="orange")
        ax[1].set_title("Validation Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        # Plot validation accuracy (x-axis: evaluation epochs, y-axis: val acc)
        ax[2].plot(eval_epochs, self.metrics["acc"], label="Validation Accuracy", color="green")
        ax[2].set_title("Validation Accuracy")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Accuracy")
        ax[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "metrics.png"))

    def train(self, epochs, eval_every, resume=False):
        """Main training loop with early stopping and periodic evaluation."""
        start_epoch = self._prepare_for_training(resume)

        acc, val_loss = None, None
        self.model.train()

        for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", initial=start_epoch, total=epochs):
            train_loss = self._train_epoch()
            self.logger.info(f"Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}")

            if (epoch + 1) % eval_every == 0:
                acc, val_loss = self._evaluate()
                self.logger.info(f"Epoch: {epoch + 1}, Accuracy: {acc:.4f}, Validation Loss: {val_loss:.4f}")

                if val_loss < self.metrics["best_val_loss"]:
                    self.metrics["best_val_loss"] = val_loss
                    self.early_stopping_counter = 0
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch+1}.pth")
                    self._save_checkpoint(epoch, checkpoint_path)
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter > self.patience:
                        self.logger.info("Early stopping")
                        break
                
                self.metrics["val_loss"].append(val_loss)
                self.metrics["acc"].append(acc)

            self.metrics["train_loss"].append(train_loss)

            # Save latest model
            self._save_checkpoint(epoch, self.latest_checkpoint, latest=True)

        # Final checkpoint if training ends without early stopping
        if val_loss and val_loss < self.metrics["best_val_loss"]:
            self._save_checkpoint(epoch)

        self._plot_metrics(eval_every)
        plot_path = os.path.join(self.exp_dir, "metrics.png")
        self.logger.info(f"Metrics plotted to {plot_path}")
        self.logger.info("\nTraining completed successfully")