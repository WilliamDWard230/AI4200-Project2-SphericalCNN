"""Resume the 100% Spherical CNN experiment through epoch 20.
Run from the project directory, alongside train.py and model.py.
"""
import csv
from pathlib import Path
import numpy as np
import torch
from torch import nn

from train import (BATCH_SIZE, DEVICE, LEARNING_RATE, MNIST_PATH,
                   NUM_EPOCHS, SEED, load_data)
from model import S2ConvNet_original

CHECKPOINT = Path('best_models/spherical_best_model.pt')
LATEST = Path('best_models/spherical_latest_model.pt')
METRICS = Path('best_models/spherical_metrics_100pct.csv')
COLUMNS = ['epoch', 'train_loss', 'test_loss', 'test_accuracy', 'train_fraction']


def main():
    if not CHECKPOINT.is_file() or not METRICS.is_file():
        raise FileNotFoundError(f'Need both {CHECKPOINT} and {METRICS}')

    with METRICS.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f'Unexpected CSV columns: {reader.fieldnames}')
        history = list(reader)
    if not history:
        raise ValueError('Metrics CSV has no rows')
    epochs = [int(row['epoch']) for row in history]
    if epochs != list(range(1, epochs[-1] + 1)):
        raise ValueError('CSV epochs must be consecutive, starting at 1')
    last_epoch = epochs[-1]
    if any(abs(float(row['train_fraction']) - 1.0) > 1e-8 for row in history):
        raise ValueError('CSV does not contain only 100% training data')

    # Only load checkpoints you created/trust.
    resume_path = LATEST if LATEST.exists() else CHECKPOINT
    checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=True)
    ckpt_epoch = int(checkpoint['epoch'])
    if ckpt_epoch != last_epoch:
        raise RuntimeError(
            f'Checkpoint is from epoch {ckpt_epoch}, but CSV ends at epoch {last_epoch}. '
            'Do not append mismatched epochs. Use a checkpoint from the last '
            'completed epoch or reconcile the CSV first.'
        )
    if abs(float(checkpoint['train_fraction']) - 1.0) > 1e-8:
        raise ValueError('Checkpoint is not for the 100% run')
    if last_epoch >= NUM_EPOCHS:
        print(f'Already at epoch {last_epoch}; nothing to do.')
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader, test_loader, train_dataset, _ = load_data(MNIST_PATH, BATCH_SIZE, 1.0)
    model = S2ConvNet_original().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_accuracy = float(checkpoint['test_accuracy'])
    print(f'Resuming from epoch {last_epoch}; training epochs {last_epoch + 1}–{NUM_EPOCHS}')
    print(f'Initial best test accuracy: {best_accuracy:.2f}%')

    for epoch in range(last_epoch + 1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            if i % 50 == 0 or i == len(train_loader):
                print(f'Epoch [{epoch}/{NUM_EPOCHS}], Iter [{i}/{len(train_loader)}] Loss: {loss.item():.4f}', flush=True)

        model.eval()
        correct = total = 0
        test_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                test_loss_sum += criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()

        accuracy = 100 * correct / total
        row = {
            'epoch': epoch,
            'train_loss': running_loss / len(train_dataset),
            'test_loss': test_loss_sum / total,
            'test_accuracy': accuracy,
            'train_fraction': 1.0,
        }
        # Append only, without rewriting existing epochs or duplicating the header.
        with METRICS.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(row)
        print(f'Epoch {epoch} | train loss={row["train_loss"]:.4f} | '
              f'test loss={row["test_loss"]:.4f} | accuracy={accuracy:.2f}%', flush=True)

        snapshot = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': accuracy,
            'train_fraction': 1.0,
            'seed': SEED,
        }
        # Always save the latest state, even if accuracy is not a new best.
        torch.save(snapshot, LATEST)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': accuracy,
                'train_fraction': 1.0,
                'seed': SEED,
            }, CHECKPOINT)
            print(f'Saved improved best checkpoint (accuracy={best_accuracy:.2f}%)', flush=True)
        else:
            print('Checkpoint unchanged (accuracy did not improve).', flush=True)


if __name__ == '__main__':
    main()
