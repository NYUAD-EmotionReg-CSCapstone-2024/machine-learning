import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.base import Transformer
from datasets.seedv.dataset import SeedVDataset

participants = [str(i) for i in range(1, 17)]
sessions = [str(i) for i in range(1, 4)]
emotions = ["happy", "sad", "fear", "neutral", "angry"]

train_set = SeedVDataset(
    root="/data/SEED-V",
    h5file="seedv.h5",
    participants=participants[:14]
)

test_set = SeedVDataset(
    root="/data/SEED-V",
    h5file="seedv.h5",
    participants=participants[14:]
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ERTNet(
#     num_classes=5,
#     num_channels=66,
#     dropout_rate=0.5,
#     kernel_length=32,
#     f1=8,
#     num_heads=4,
#     d=4,
#     f2=16
# ).to(device)
model = Transformer(
    n_channels=62,
    embed_dim=64,
    num_heads=4,
    num_classes=5
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# use tqdm for progress bar
from tqdm import tqdm

n_epochs = 100
loss_values = []
avg_loss_values = []
acc_values = []

# Assuming n_epochs, model, criterion, optimizer, train_loader, and test_loader are defined
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    
    # Wrapping the train_loader with tqdm to show progress
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch") as pbar:
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            running_loss += loss.item()
            
            # Update the progress bar with the loss
            pbar.set_postfix({"loss": running_loss / (i + 1)})  # Average loss for the current epoch
            pbar.update(1)  # Increment the progress bar

    # Print the average loss for the epoch
    avg_loss_value = running_loss / len(train_loader)
    avg_loss_values.append(avg_loss_value)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss_value:.4f}")

    # Evaluate the model on the test set
    acc = evaluate(model, test_loader)
    acc_values.append(acc)
    print(f"Accuracy: {acc:.4f}")

print('Finished Training')

# save the model
torch.save(model.state_dict(), "seedv_transformer.pth")

# plot the loss and acc values and save the plot
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
plt.savefig("training_plot.png")