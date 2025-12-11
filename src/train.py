"""
Main training script for network attack detection.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.helpers import (
    load_config,
    save_checkpoint,
    evaluate_model,
    print_metrics,
    set_seed
)
from src.models.fusion_net import FusionNet, SingleSourceNet


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        # Handle both fusion and single-source models
        if len(batch) == 3:
            traffic, logs, labels = batch
            traffic, logs, labels = traffic.to(device), logs.to(device), labels.to(device)
            outputs = model(traffic, logs)
        else:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                traffic, logs, labels = batch
                traffic, logs, labels = traffic.to(device), logs.to(device), labels.to(device)
                outputs = model(traffic, logs)
            else:
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = evaluate_model(all_labels, all_preds)

    return avg_loss, metrics


def main(config_path: str):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    set_seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: Load your data here
    # Example placeholder - replace with actual data loading
    print("\n" + "=" * 50)
    print("Please load your dataset!")
    print("Place your data files in: data/raw/")
    print("=" * 50)

    # Initialize TensorBoard
    writer = SummaryWriter(config['output']['results_dir'])

    print("\nTraining setup complete. Ready to train once data is loaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train network attack detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    main(args.config)
