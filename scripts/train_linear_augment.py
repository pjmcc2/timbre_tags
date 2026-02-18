
# AI-improved Code
# train_projection.py
import argparse
import json
import math
import os
import random
import pickle
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.torch_classes import embDataset, Projection
from src.calc_metrics import calc_dist_metrics, calc_rep_metrics


def set_seed(seed: int = 1066):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embeddings(clotho_path, sb_path):
    with open(clotho_path, "rb") as f:
        c_t_embs,c_a_embs = pickle.load(f)

    with open(sb_path, "rb") as f:
        sb_t_embs,sb_a_embs = pickle.load(f)
   

    total_t_embs = np.vstack([c_t_embs, sb_t_embs]).astype(np.float32)
    total_a_embs = np.vstack([c_a_embs, sb_a_embs]).astype(np.float32)
    return total_t_embs, total_a_embs


def maybe_normalize(x: torch.Tensor, eps: float = 1e-8):
    # L2 normalize row-wise
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)


def train_model(
    clotho_path,
    sb_path,
    batch_size: int = 128,
    epochs: int = 250,
    lr: float = 1e-3,
    noise_std: float | None = 0.023023764,
    seed: int = 1066,
    compute_training_stats: bool = False,
    normalize_for_metrics: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    set_seed(seed)

    # Load data
    total_t_embs, total_a_embs = load_embeddings(
        clotho_path,sb_path
    )

    # Dataset / DL
    emb_dataset = embDataset(total_t_embs, total_a_embs)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model — ensure out_dim matches audio embedding size
    in_dim = total_t_embs.shape[1]
    out_dim = total_a_embs.shape[1]
    model = Projection(in_dim, out_dim).to(device)
    model.train()

    data_loader = torch.utils.data.DataLoader(
        emb_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory if device == "cuda" else False,
        generator=torch.Generator().manual_seed(seed),
    )

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    epoch_logs = []
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0

        for X, y in data_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if noise_std is not None and noise_std > 0:
                X_aug = X + noise_std * torch.randn_like(X)
            else:
                X_aug = X

            optim.zero_grad(set_to_none=True)
            outputs = model(X_aug)
            loss = loss_fn(outputs, y)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)

        log_row = {"epoch": epoch + 1, "train_loss": avg_loss}

        if compute_training_stats:
            # Compute metrics on the full dataset (single pass, no grad)
            model.eval()
            with torch.no_grad():
                # Accumulate over full dataset
                all_X, all_out, all_y = [], [], []
                for Xb, yb in data_loader:
                    Xb = Xb.to(device)
                    ob = model(Xb)
                    all_X.append(Xb)
                    all_out.append(ob)
                    all_y.append(yb.to(device))

                X_full = torch.cat(all_X, dim=0)
                out_full = torch.cat(all_out, dim=0)
                y_full = torch.cat(all_y, dim=0)

                if normalize_for_metrics:
                    Xm = maybe_normalize(X_full)
                    Om = maybe_normalize(out_full)
                    Ym = maybe_normalize(y_full)
                else:
                    Xm, Om, Ym = X_full, out_full, y_full

                # Move to CPU numpy if metrics expect numpy
                dm = calc_dist_metrics(Xm.detach().cpu().numpy(),
                                       Om.detach().cpu().numpy())
                rm = calc_rep_metrics(Xm.detach().cpu().numpy(),
                                      Om.detach().cpu().numpy(),
                                      Ym.detach().cpu().numpy(),
                                      supervised=True)

                # Merge dict-like metrics; if they return tuples, adapt here
                if isinstance(dm, dict):
                    for k, v in dm.items():
                        log_row[f"dist_{k}"] = float(v)
                else:
                    log_row["dist_metrics"] = str(dm)

                if isinstance(rm, dict):
                    for k, v in rm.items():
                        log_row[f"rep_{k}"] = float(v)
                else:
                    log_row["rep_metrics"] = str(rm)

            model.train()

        epoch_logs.append(log_row)

    return model, pd.DataFrame(epoch_logs)


def save_model_state(model: torch.nn.Module, out_path: str, overwrite: bool = False):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"{out_path} exists. Use overwrite=True to replace it."
        )
    torch.save(model.state_dict(), out_path)


def main():
    parser = argparse.ArgumentParser(description="Train a projection model on embeddings.")
    parser.add_argument("--clotho_path", default="data/processed/clotho/clotho_clap_embeddings.pickle")
    parser.add_argument("--sb_path", default="data/processed/clotho/sb_clap_embeddings.pickle")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=0.023023764, help="Gaussian noise std for input augmentation.")
    parser.add_argument("--seed", type=int, default=1066)
    parser.add_argument("--compute_stats", action="store_true")
    parser.add_argument("--metrics_no_norm", action="store_true", help="Disable L2-normalization for metrics.")
    parser.add_argument("--model_out", default=None, help="Path to save model state_dict (.pt).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_csv", default=None, help="Path to save training logs CSV.")
    parser.add_argument("--loss_plot", default=None, help="Path to save training loss plot PNG.")
    args = parser.parse_args()

    model, logs_df = train_model(
        clotho_path=args.clotho_path,

        sb_path=args.sb_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        noise_std=(args.noise_std if args.noise_std > 0 else None),
        seed=args.seed,
        compute_training_stats=args.compute_stats,
        normalize_for_metrics=not args.metrics_no_norm,
    )

    if args.model_out:
        save_model_state(model, args.model_out, overwrite=args.overwrite)

    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        logs_df.to_csv(args.log_csv, index=False)

    if args.loss_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(logs_df["epoch"], logs_df["train_loss"], label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        os.makedirs(os.path.dirname(args.loss_plot) or ".", exist_ok=True)
        plt.savefig(args.loss_plot, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()