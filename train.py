import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import subprocess
from factories import ModelFactory, EncoderFactory, OptimizerFactory, SchedulerFactory, DatasetFactory, SplitterFactory
from trainers import Trainer
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

def launch_tensorboard(log_dir):
    """Launch TensorBoard as a background process and open it in a web browser."""
    print(f"Starting TensorBoard from {log_dir}")
    try:
        # Use --logdir_spec to provide better naming for runs
        parent_dir = os.path.dirname(log_dir)
        
        tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", parent_dir], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        return tensorboard_process
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")
        return None

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

    use_config = config.get("model", {}).get("encoder", None)
    if use_config:
        model = EncoderFactory.wrap(
            use_config["name"], 
            model, 
            **use_config.get("params", {})
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

    model_name = config["model"]["name"]
    encoder_name = use_config["name"] if use_config else None

    if encoder_name:
        model_filename = f"{encoder_name}_{model_name}_trained_model.pth"
    else:
        model_filename = f"{model_name}_trained_model.pth"
    print("Output Model Filename:", model_filename)

    # Added these pre-calculated weights to give more loss penalization towards classes: [negative, neutral, positive]
    # Test 1 Weights for balancing classes [negative, neutral, positive] = torch.tensor([0.523430585861206, 1.6582633256912231, 2.055555582046509])
    #   - This is due to the imbalance - more negative chunks in the dataset than neutral and positive

    # Test 2 
    loss_weights = torch.tensor([0.523430585861206, 1.6582633256912231, 2.055555582046509], dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        exp_dir=exp_dir,
        model_filename=model_filename
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
        device=device,
        **config["dataset"]["params"], 
        load=args.load
    )
    
    splitter = SplitterFactory.create(
        config["splitter"]["name"], 
        dataset=dataset, 
        **config["splitter"]["params"]
    )

    # Launch TensorBoard if requested
    tensorboard_process = None
    if args.board:
        # Point TensorBoard to the experiment root directory 
        # instead of a specific run's tensorboard dir
        # This allows TensorBoard to show all runs with proper names
        tensorboard_dir = config["exp_dir"]
        tensorboard_process = launch_tensorboard(tensorboard_dir)
        if tensorboard_process:
            print("TensorBoard running at http://localhost:6006")
            print(f"Experiment directory: {tensorboard_dir}")

    # Handle K-Fold Cross-Validation
    if config["splitter"]["name"] == "kfold":
        num_folds = config["splitter"]["params"]["k"]
        fold_metrics = []

        for fold_idx in range(num_folds):
            print(f"\nStarting Fold {fold_idx + 1}/{num_folds}")
            trainer = create_trainer(config, splitter, device, fold_idx)
            
            trainer.train(
                epochs=config["epochs"], 
                eval_every=config["eval_every"], 
                patience=config["patience"], 
                resume=args.resume
            )
            
            fold_metrics.append(trainer.metrics)

        avg_val_loss = sum([m["best_val_loss"] for m in fold_metrics]) / num_folds
        print(f"\nK-Fold Cross-Validation Completed\nAverage Validation Loss: {avg_val_loss:.4f}")
   
    else:
        # Single train-test split
        trainer = create_trainer(config, splitter, device)
        
        trainer.train(
            epochs=config["epochs"],
            eval_every=config["eval_every"], 
            patience=config["patience"], 
            resume=args.resume
        )
    
    # Keep TensorBoard running until user manually closes it
    if tensorboard_process:
        try:
            print("TensorBoard is running. Press Ctrl+C to exit.")
            tensorboard_process.wait()
        except KeyboardInterrupt:
            print("Terminating TensorBoard...")
            tensorboard_process.terminate()
            tensorboard_process.wait()
            print("TensorBoard terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the SEED-V dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (without .yaml extension)")
    parser.add_argument("--load", action="store_true", help="Load the dataset in memory for faster training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--board", action="store_true", help="Launch TensorBoard server to monitor training")
    args = parser.parse_args()
    main(args)