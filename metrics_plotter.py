import pickle
import matplotlib.pyplot as plt

EXP_NUM = 2

EXP_DIR = f'./experiments/exp_{EXP_NUM}'

metrics_file_path = f'{EXP_DIR}/metrics.pkl'

with open(metrics_file_path, 'rb') as f:
    metrics = pickle.load(f)
    loss_values, avg_loss_values, acc_values = metrics['loss'], metrics['avg_loss'], metrics['accuracy']

    # create three plots, one loss values, one avg_loss values, and one accuracy values
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
    plt.savefig(f"{EXP_DIR}/plot.png")