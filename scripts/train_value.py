# scripts/train_value.py

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml.parse_pgn import generate_value_samples_from_pgn
from ml.models import ChessValueNet


class ValueDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of (pos_tensor, value)
        pos_tensor: np.ndarray [C,8,8]
        value: float [-1,1]
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pos, val = self.samples[idx]
        x = torch.from_numpy(pos)          # [C,8,8]
        y = torch.tensor(val, dtype=torch.float32)  # scalar
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=str, required=True,
                        help="Path to PGN file (already downloaded).")
    parser.add_argument("--max_positions", type=int, default=50000,
                        help="Limit number of positions to sample.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="data/value_net.pt")
    args = parser.parse_args()

    pgn_path = Path(args.pgn)
    assert pgn_path.exists(), f"PGN file {pgn_path} does not exist."

    print(f"Collecting up to {args.max_positions} positions from {pgn_path}...")
    samples = []
    for pos_tensor, value in generate_value_samples_from_pgn(pgn_path, max_positions=args.max_positions):
        samples.append((pos_tensor, value))
    print(f"Collected {len(samples)} positions.")

    # Simple train/val split
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_ds = ValueDataset(train_samples)
    val_ds = ValueDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ChessValueNet(in_channels=18).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)   # [B]
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                total_val_loss += loss.item() * x.size(0)

        val_loss = total_val_loss / len(val_ds)
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
