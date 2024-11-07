import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

# from models.conv_tansformer import BaseModel
# from models.base_model import BaseModel as Model
from models.base_transformer import BaseTransformer as Model

from datasets.seedv.dataset import SeedVDataset

# CONSTANTS
EPOCHS = 100
TRAIN_SET_SIZE = 14
TEST_SET_SIZE = 2
BATCH_SIZE = 512
LEARNING_RATE = 2.25e-3
EVAL_EVERY = 5
SAVE_EVERY = 5

# Model Parameters
N_SAMPLES = 200
N_CHANNELS = 62
N_HEADS = 8
N_LAYERS = 4
N_CLASSES = 5

EXP_NUM = 3
EXP_DIR = f"./experiments/exp_{EXP_NUM}"
CHECKPOINT_DIR = f"{EXP_DIR}/checkpoints"

# Setup
participants = [str(i) for i in range(1, 17)]
sessions = [str(i) for i in range(1, 4)]
emotions = ["happy", "sad", "fear", "neutral", "angry"]

train_set = SeedVDataset(
    root="/data/SEED-V",
    h5file="seedv.h5",
    participants=participants[:TRAIN_SET_SIZE]
)

test_set = SeedVDataset(
    root="/data/SEED-V",
    h5file="seedv.h5",
    participants=participants[:-TEST_SET_SIZE]
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)  # Move data to device
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Model(
#     n_samples=N_SAMPLES,
#     n_channels=N_CHANNELS,
#     n_heads=N_HEADS,
#     n_layers=N_LAYERS,
#     n_classes=N_CLASSES,
# ).to(device)

model = Model(
    n_samples=N_SAMPLES,
    n_channels=N_CHANNELS,
    n_classes=N_CLASSES,
).to(device)

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# dump info
with open(f"{EXP_DIR}/model_architecture.txt", "w") as f:
    # model summary
    f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    f.write("\n\n")
    f.write(f"Model: {model}")
    f.write("\n\n")
    # training parameters
    f.write(f"Number of Samples (chunk size): {N_SAMPLES}")
    f.write(f"Number of Channels: {N_CHANNELS}")
    f.write(f"Epochs: {EPOCHS}")
    f.write(f"Train Set Size: {TRAIN_SET_SIZE}")
    f.write(f"Test Set Size: {TEST_SET_SIZE}")
    f.write(f"Batch Size: {BATCH_SIZE}")
    f.write(f"Learning Rate: {LEARNING_RATE}")
    f.write(f"Evaluation Frequency: {EVAL_EVERY}")
    f.write(f"Save Frequency: {SAVE_EVERY}")
    f.write("/n")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
loss_values = []
avg_loss_values = []
acc_values = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch") as pbar:
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            running_loss += loss.item()
            
            pbar.set_postfix({"loss": running_loss / (i + 1)})
            pbar.update(1)

    avg_loss_value = running_loss / len(train_loader)
    avg_loss_values.append(avg_loss_value)
    print(f"Average Loss: {avg_loss_value:.4f}")

    # Evaluate the model every 5 epochs
    if (epoch + 1) % EVAL_EVERY == 0:
        acc = evaluate(model, test_loader)
        acc_values.append(acc)
        print(f"Accuracy: {acc:.4f}")

    # Save model checkpoint every 5 epochs
    if (epoch + 1) % SAVE_EVERY == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {CHECKPOINT_DIR}/model_epoch_{epoch + 1}.pth")

print('Finished Training')

# Save the final model
torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/seedv_transformer_final.pth")

# Plot the loss and accuracy values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_values, label="Training Loss")
plt.plot(avg_loss_values, label="Average Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_values, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(f"{EXP_DIR}/plot.png")

# Save the losses and accuracies
import pickle

with open(f"{EXP_DIR}/metrics.pkl", "wb") as f:
    pickle.dump({
        "loss": loss_values,
        "avg_loss": avg_loss_values,
        "accuracy": acc_values
    }, f)
