import os
import shutil
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Trainer(ABC):
    def __init__(self, train_loader, val_loader, model, loss_fn, optimizer, device, exp_dir, model_filename, run_name=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = None
        self.exp_dir = exp_dir
        self.model_filename = model_filename
        self.device = device

        # Set run name with proper format for regex searching
        if run_name is None:
            # Extract from exp_dir with timestamp to ensure uniqueness
            base_name = os.path.basename(exp_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"run_{base_name}_{timestamp}"
        else:
            # Ensure run name follows a consistent pattern
            if not run_name.startswith("run_"):
                self.run_name = f"run_{run_name}"
            else:
                self.run_name = run_name
        
        self._setup_directories(exp_dir)
        self.early_stopping_counter = 0
        self.metrics = {
            "best_val_loss": float("inf"),
            "train_loss": [],
            "val_loss": [],
            "acc": []
        }
        
        # Initialize TensorBoard writer with structured log directory
        self.tb_log_dir = os.path.join(exp_dir, "tensorboard", self.run_name)
        os.makedirs(self.tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_log_dir)
        
        # Log run name for easy identification
        with open(os.path.join(exp_dir, "run_name.txt"), "w") as f:
            f.write(self.run_name)
        
        self._setup_tensorboard()

    def set_scheduler(self, scheduler):
        """Add scheduler to trainer after initialization"""
        self.scheduler = scheduler

    def _setup_logger(self, exp_dir, mode="w"):
        """Set up the logger to log training information."""
        self.logger = logging.getLogger(__name__)

        # Remove any previous existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.setLevel(logging.INFO)
        
        # File handler for logging to file
        log_file = os.path.join(exp_dir, "train.log")
        file_handler = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for logging to console
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

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
        
        # Recreate TensorBoard directory with proper run name
        self.tb_log_dir = os.path.join(exp_dir, "tensorboard", self.run_name)
        os.makedirs(self.tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_log_dir)
        
        # Log run name for easy identification
        with open(os.path.join(exp_dir, "run_name.txt"), "w") as f:
            f.write(self.run_name)
            
        self._setup_tensorboard()

    def _log_model_info(self):
        """Log model, loss function, and optimizer info."""
        self.logger.info("Model:\n{}".format(self.model))
        self.logger.info("Loss Function:\n{}".format(self.loss_fn))
        self.logger.info("Optimizer:\n{}".format(self.optimizer))
        
        # Log model architecture to TensorBoard
        try:
            # Get a sample input for model graph visualization
            sample_data = next(iter(self.train_loader))[0][:1].to(self.device)
            self.writer.add_graph(self.model, sample_data)
            self.logger.info("Model architecture added to TensorBoard")
        except Exception as e:
            self.logger.warning(f"Failed to add model graph to TensorBoard: {e}")

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
        self.logger.debug(f"Checkpoint saved to {checkpoint_path}")

    def _save_full_model(self, model_path=None):
        """Save the entire model (not just state_dict) for easy deployment"""
        if model_path is None:
            model_path = os.path.join(self.exp_dir, self.model_filename)
        
        # Set model to eval mode before saving
        self.model.eval()
        
        # Save the complete model
        torch.save(self.model, model_path)
        self.logger.info(f"Full model saved to {model_path}")
        
        return model_path

    def _evaluate(self):
        """Evaluate model performance on the validation set."""
        self.model.eval()
        correct = total = 0
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc="Validation Iterations", leave=False) as pbar:
                for idx, (data, labels) in enumerate(self.val_loader):
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = F.softmax(self.model(data), dim=1) # manually apply softmax
                    loss = self.loss_fn(outputs, labels)
                    batch_loss = loss.item()
                    val_loss += batch_loss
                    
                    # Log validation batch loss to TensorBoard
                    self.writer.add_scalar('batch/validation_loss', batch_loss, 
                                          self.current_epoch * len(self.val_loader) + idx)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    
                    # Store predictions and labels for confusion matrix
                    all_preds.append(predicted.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    
                    pbar.set_postfix({"val_loss": batch_loss})
                    pbar.update()
                    
        acc = correct / total
        val_loss = val_loss / len(self.val_loader)
        
        # Combine all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        return acc, val_loss, all_preds, all_labels

    def _train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        self.current_epoch = epoch  # Store current epoch for validation logging
        
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.total_epochs}", leave=False) as pbar:
            for idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                # Log batch loss to TensorBoard with better naming
                global_step = epoch * len(self.train_loader) + idx
                self.writer.add_scalar('batch/training_loss', loss.item(), global_step)
                
                pbar.set_postfix({"loss": running_loss / (idx + 1)})
                pbar.update()

        # Step the scheduler after each epoch
        if self.scheduler is not None:
            self.scheduler.step()
            # Log learning rate to TensorBoard
            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('optimizer/learning_rate', current_lr, epoch)
            
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
                self._setup_tensorboard()
        else:
            self._cleanup_dir(self.exp_dir)
            self._setup_logger(self.exp_dir, mode="w")
            self._log_model_info()
            self._setup_tensorboard()
        
        self.current_epoch = start_epoch  # Initialize current epoch tracker
        return start_epoch
    
    def _plot_metrics(self, eval_every):
        """Plot and save training metrics."""
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        eval_epochs = list(range(eval_every, len(self.metrics["val_loss"]) * eval_every + 1, eval_every))

        # Plot training loss
        ax[0].plot(self.metrics["train_loss"], label="Train Loss")
        ax[0].set_title("Training Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # Plot validation loss
        ax[1].plot(eval_epochs, self.metrics["val_loss"], label="Validation Loss", color="orange")
        ax[1].set_title("Validation Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        # Plot validation accuracy
        ax[2].plot(eval_epochs, self.metrics["acc"], label="Validation Accuracy", color="green")
        ax[2].set_title("Validation Accuracy")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Accuracy")
        ax[2].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.exp_dir, "metrics.png")
        plt.savefig(plot_path)
        plt.close(fig)
        
        # Add the figure to TensorBoard
        self.writer.add_figure('Training Metrics', fig, close=False)
        self.logger.info(f"Metrics plotted to {plot_path} and added to TensorBoard")

    def _setup_tensorboard(self):
        """Setup TensorBoard with custom layout for better organization."""
        layout = {
            "Epoch Metrics": {
                "Accuracy": ["Multiline", ["metrics/accuracy"]],
                "Losses": ["Multiline", ["epoch/training_loss", "epoch/validation_loss"]]
            },
            "Batch Monitoring": {
                "Training Loss": ["Multiline", ["batch/training_loss"]],
                "Validation Loss": ["Multiline", ["batch/validation_loss"]]
            },
            "Optimizer Stats": {
                "Learning Rate": ["Multiline", ["optimizer/learning_rate"]]
            }
        }
        
        self.writer.add_custom_scalars(layout)
        # Log run name as a text in TensorBoard for easy identification
        self.writer.add_text('Run Info', f"Run Name: {self.run_name}", 0)

    def train(self, epochs, eval_every, patience, resume=False):
        """Main training loop with early stopping and periodic evaluation."""

        assert eval_every <= epochs, \
            "Evaluation frequency must be less than total epochs"
        assert patience >= eval_every, \
            "Patience must be greater than or equal to evaluation frequency"
        assert patience % eval_every == 0, \
            "Patience must be divisible by evaluation frequency"

        start_epoch = self._prepare_for_training(resume)
        self.total_epochs = epochs  # Store total epochs for progress tracking
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Run name: {self.run_name}")
        self.logger.info(f"Evaluation every {eval_every} epochs, patience: {patience} epochs")
        self.logger.info(f"Device: {self.device}")
        
        # Add hyperparameters to TensorBoard
        self.writer.add_text('Hyperparameters', 
                           f"Run Name: {self.run_name}\n"
                           f"Epochs: {epochs}\n"
                           f"Eval Frequency: {eval_every}\n"
                           f"Patience: {patience}", 0)

        acc, val_loss = None, None
        best_acc = 0.0
        
        start_time = datetime.now()
        self.logger.info(f"Training started at: {start_time}")

        for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress", initial=start_epoch, total=epochs):
            epoch_start_time = datetime.now()
            
            # Training phase
            train_loss = self._train_epoch(epoch)
            self.metrics["train_loss"].append(train_loss)
            self.writer.add_scalar('epoch/training_loss', train_loss, epoch)
            
            self.logger.info(f"Epoch: {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}")

            # Evaluation phase
            if (epoch + 1) % eval_every == 0:
                acc, val_loss, all_preds, all_labels = self._evaluate()
                self.metrics["val_loss"].append(val_loss)
                self.metrics["acc"].append(acc)
                
                # Log metrics to TensorBoard with better naming
                self.writer.add_scalar('epoch/validation_loss', val_loss, epoch)
                self.writer.add_scalar('metrics/accuracy', acc, epoch)
                
                # Add confusion matrix if we have a reasonable number of classes
                if len(np.unique(all_labels)) <= 20:  # Limit to avoid huge matrices
                    try:
                        from sklearn.metrics import confusion_matrix
                        import seaborn as sns # type: ignore
                        
                        cm = confusion_matrix(all_labels, all_preds)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted labels')
                        ax.set_ylabel('True labels')
                        ax.set_title(f'Confusion Matrix - Epoch {epoch+1}')
                        
                        self.writer.add_figure('confusion_matrix', fig, epoch)
                    except ImportError:
                        self.logger.warning("sklearn or seaborn not available for confusion matrix")
                
                epoch_time = datetime.now() - epoch_start_time
                self.logger.info(f"Epoch: {epoch + 1}/{epochs}, "
                                f"Accuracy: {acc:.4f}, Validation Loss: {val_loss:.4f}, "
                                f"Time: {epoch_time}")

                # Early stopping check
                if val_loss < self.metrics["best_val_loss"]:
                    improvement = (self.metrics["best_val_loss"] - val_loss) / self.metrics["best_val_loss"] * 100
                    self.metrics["best_val_loss"] = val_loss
                    self.early_stopping_counter = 0
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch+1}.pth")
                    self._save_checkpoint(epoch, checkpoint_path)
                    
                    self.logger.info(f"Validation loss improved by {improvement:.2f}%, saving model to {checkpoint_path}")
                    
                    # Save best model if accuracy improved
                    if acc > best_acc:
                        best_acc = acc
                        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                        self._save_checkpoint(epoch, best_model_path)
                        self.logger.info(f"Best accuracy so far: {best_acc:.4f}, saving to {best_model_path}")
                else:
                    self.early_stopping_counter += eval_every
                    self.logger.info(f"Validation loss did not improve. Early stopping counter: "
                                    f"{self.early_stopping_counter}/{patience}")
                    
                    if self.early_stopping_counter > patience:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break

            # Save latest model
            self._save_checkpoint(epoch, self.latest_checkpoint, latest=True)

        # Final checkpoint if training ends without early stopping
        if val_loss and val_loss < self.metrics["best_val_loss"]:
            final_checkpoint_path = os.path.join(self.checkpoint_dir, "final_checkpoint.pth")
            self._save_checkpoint(epoch, final_checkpoint_path)
            self.logger.info(f"Training complete. Final model saved to {final_checkpoint_path}")
            
        # Plot metrics
        self._plot_metrics(eval_every)

        # Save the full model
        saved_model_path = self._save_full_model()
        self.logger.info(f"Saved full model to {saved_model_path}")
                
        total_time = datetime.now() - start_time
        self.logger.info(f"\nTraining completed successfully in {total_time}")
        self.logger.info(f"Best validation loss: {self.metrics['best_val_loss']:.4f}, "
                        f"Best accuracy: {best_acc:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()