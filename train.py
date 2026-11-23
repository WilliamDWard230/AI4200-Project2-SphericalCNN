
import numpy as np
import os
import torch

torch.set_num_threads(
    int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
)
import torch.utils.data as data_utils
import gzip
import pickle
from model import S2ConvNet_original
from torch import nn

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

MNIST_PATH = "s2_mnist.gz"
CHECKPOINT_PATH = os.path.join("best_models", "spherical_best_model.pt")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(path, batch_size):

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

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
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
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset



def main():

    train_loader, test_loader, train_dataset, _ = load_data(
        MNIST_PATH, BATCH_SIZE)

    classifier = S2ConvNet_original()
    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
                epoch+1, NUM_EPOCHS, i+1, len(train_dataset)//BATCH_SIZE,
                loss.item()), end="")
        print("")
        correct = 0
        total = 0
        for images, labels in test_loader:

            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = classifier(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).long().sum().item()

        accuracy = 100 * correct / total
        print('Test Accuracy: {0:.2f}%'.format(accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_accuracy": accuracy,
            }, CHECKPOINT_PATH)
            print(f"Saved best model to {CHECKPOINT_PATH} (accuracy: {best_accuracy:.2f}%)")


if __name__ == '__main__':

    main()
