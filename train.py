"""S2CNN MNIST training with logging and checkpointing.

Outputs (all under --out-dir):
    train.log     human-readable log (also echoed to stdout)
    metrics.jsonl one JSON line per epoch (easy to load with pandas / plot)
    latest.pt     full training state, overwritten after every epoch (for resuming)
    best.pt       full training state at the epoch with the best test accuracy

If latest.pt exists, training resumes from it automatically (handy for SLURM
requeues / preemption). Pass --fresh to ignore it and start over.
"""
import argparse
import gzip
import json
import logging
import math
import os
import pickle
import socket
import sys
import time

import numpy as np
import torch

torch.set_num_threads(
    int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
)
import torch.utils.data as data_utils  # noqa: E402
from torch import nn  # noqa: E402

from model import S2ConvNet_original  # noqa: E402

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
KEEP = 1.0  # NOTE: load_data() currently ignores keep_amount
LOG_INTERVAL = 100  # iterations between training log lines

MNIST_PATH = "s2_mnist.gz"
OUT_DIR = "runs/s2cnn_mnist"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(out_dir):
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:  # already configured
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    # Append mode so a resumed run continues the same log file.
    logfile = logging.FileHandler(os.path.join(out_dir, "train.log"), mode="a")
    logfile.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(logfile)


def append_metrics(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model, optimizer, epochs_done, best_acc, args):
    """Atomically write the full training state (write to tmp, then rename)."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epochs_done,  # number of *completed* epochs
        "best_acc": best_acc,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "config": vars(args),
    }
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX, so a crash can't corrupt `path`


def load_checkpoint(path, model, optimizer):
    """Restore state in-place; returns (epochs_done, best_acc)."""
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    # Restore RNG state so shuffling / dropout continue as if uninterrupted.
    torch.set_rng_state(ckpt["torch_rng"].cpu())
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
        except Exception as e:  # e.g. different number of GPUs than before
            logger.warning("Could not restore CUDA RNG state: %s", e)

    return ckpt["epoch"], ckpt["best_acc"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path, batch_size, keep_amount=1.0):

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
    # No need to shuffle for evaluation.
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, epoch, num_epochs, log_interval):
    model.train()
    num_batches = len(loader)
    epoch_loss, epoch_seen = 0.0, 0
    window_loss, window_seen = 0.0, 0
    window_start = time.time()

    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            raise FloatingPointError(
                f"Non-finite loss ({loss_val}) at epoch {epoch + 1}, iter {i + 1}"
            )

        bs = labels.size(0)
        epoch_loss += loss_val * bs
        epoch_seen += bs
        window_loss += loss_val * bs
        window_seen += bs

        if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
            elapsed = time.time() - window_start
            logger.info(
                "Epoch [%d/%d] Iter [%d/%d] loss %.4f (avg over last %d samples) | %.1f it/s",
                epoch + 1, num_epochs, i + 1, num_batches,
                window_loss / window_seen, window_seen,
                (window_seen / bs) / max(elapsed, 1e-9),
            )
            window_loss, window_seen = 0.0, 0
            window_start = time.time()

    return epoch_loss / epoch_seen


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=MNIST_PATH)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--log-interval", type=int, default=LOG_INTERVAL,
                   help="iterations between training log lines")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing latest.pt and start from scratch")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(args.out_dir)

    latest_path = os.path.join(args.out_dir, "latest.pt")
    best_path = os.path.join(args.out_dir, "best.pt")
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")

    logger.info("=" * 70)
    logger.info("Host: %s | SLURM job: %s", socket.gethostname(),
                os.environ.get("SLURM_JOB_ID", "n/a"))
    logger.info("torch %s | device %s%s | threads %d", torch.__version__, DEVICE,
                f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "",
                torch.get_num_threads())
    logger.info("Config: %s", json.dumps(vars(args), sort_keys=True))

    train_loader, test_loader, train_dataset, test_dataset = load_data(
        args.data, args.batch_size)
    logger.info("Data: %d train / %d test samples", len(train_dataset), len(test_dataset))

    classifier = S2ConvNet_original()
    classifier.to(DEVICE)
    logger.info("#params %d", sum(x.numel() for x in classifier.parameters()))

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    start_epoch, best_acc = 0, 0.0
    if os.path.exists(latest_path) and not args.fresh:
        start_epoch, best_acc = load_checkpoint(latest_path, classifier, optimizer)
        logger.info("Resumed from %s (%d epochs done, best test acc %.2f%%)",
                    latest_path, start_epoch, best_acc)
    else:
        logger.info("Starting fresh run")

    if start_epoch >= args.epochs:
        logger.info("Nothing to do: %d epochs already completed (target %d)",
                    start_epoch, args.epochs)
        return

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(classifier, train_loader, criterion, optimizer,
                                     epoch, args.epochs, args.log_interval)
        train_time = time.time() - t0

        test_loss, test_acc = evaluate(classifier, test_loader, criterion)
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        logger.info(
            "Epoch [%d/%d] done | train_loss %.4f | test_loss %.4f | test_acc %.2f%%"
            " | best %.2f%% | train time %.1fs | total %.1fs",
            epoch + 1, args.epochs, train_loss, test_loss, test_acc, best_acc,
            train_time, time.time() - t0,
        )

        append_metrics(metrics_path, {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_acc": best_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "train_time_s": train_time,
            "timestamp": time.time(),
        })

        save_checkpoint(latest_path, classifier, optimizer, epoch + 1, best_acc, args)
        if is_best:
            save_checkpoint(best_path, classifier, optimizer, epoch + 1, best_acc, args)
            logger.info("New best model saved to %s", best_path)

    logger.info("Training complete. Best test accuracy: %.2f%%", best_acc)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Make sure the traceback lands in train.log, not just SLURM's stderr.
        logger.exception("Training crashed")
        raise
