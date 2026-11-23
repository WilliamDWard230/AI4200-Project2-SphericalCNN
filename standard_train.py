
import argparse
import csv
import numpy as np
import os
import torch

torch.set_num_threads(
    int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
)
import torch.utils.data as data_utils
import gzip
import pickle
from standard_model import StandardCNN
from torch import nn

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

MNIST_PATH = "s2_mnist.gz"
SEED = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(path, batch_size, train_fraction):

    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))

    # TODO normalize dataset
    mean = train_data.mean()
    stdv = train_data.std()

    train_data -= mean
    train_data /= stdv

    full_train_dataset = data_utils.TensorDataset(train_data, train_labels)
    # A fixed permutation means 25% is contained within 50% and both
    # models use exactly the same training examples.
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(full_train_dataset), generator=generator)
    subset_size = max(1, int(len(indices) * train_fraction))
    train_dataset = data_utils.Subset(full_train_dataset, indices[:subset_size].tolist())
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    mean = test_data.mean()
    stdv = test_data.std()

    test_data -= mean
    test_data /= stdv

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Fraction of training data to use (e.g. 0.25, 0.5, 1.0)")
    args = parser.parse_args()
    if not (0 < args.train_fraction <= 1):
        parser.error("--train-fraction must be > 0 and <= 1")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pct = round(args.train_fraction * 100)
    # Keep the existing full-data checkpoint filename unchanged.
    suffix = "" if args.train_fraction == 1.0 else f"_{pct}pct"
    checkpoint_path = os.path.join("best_models", f"standard_best_model{suffix}.pt")
    metrics_path = os.path.join("best_models", f"standard_metrics{suffix or '_100pct'}.csv")

    train_loader, test_loader, train_dataset, test_dataset = load_data(
        MNIST_PATH, BATCH_SIZE, args.train_fraction)
    print(f"Training samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    classifier = StandardCNN()
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_accuracy = -1.0

    history = []
    for epoch in range(NUM_EPOCHS):
        classifier.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item() * labels.size(0)

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch+1, NUM_EPOCHS, i+1, len(train_loader),
                loss.item()), end="")
        print("")
        classifier.eval()
        correct = 0
        total = 0
        test_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = classifier(images)
                test_loss_sum += criterion(outputs, labels).item() * labels.size(0)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        history.append({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_dataset),
            "test_loss": test_loss_sum / total,
            "test_accuracy": accuracy,
            "train_fraction": args.train_fraction,
        })
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print('Test Accuracy: {0:.2f}%'.format(accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_accuracy": accuracy,
                "train_fraction": args.train_fraction,
                "seed": SEED,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path} (accuracy: {best_accuracy:.2f}%)")


if __name__ == '__main__':

    main()
