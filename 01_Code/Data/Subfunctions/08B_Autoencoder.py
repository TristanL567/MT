"""
06C_Autoencoder.py
==================
Beta-VAE with optional supervised classification loss for CSI prediction.

Architecture:
    Encoder : 463 → 256 → 128 → 64 → (z_mean, z_log_var)
    Decoder : z → 64 → 128 → 256 → 463
    Classifier head : z_mean → 16 → 1  (optional, gamma > 0)

Loss:
    total = MSE_recon + beta * KL + gamma * BCE_classification

Inputs  (from R pipeline):
    - features_raw.rds   → loaded via pyreadr
    - splits.rds         → train/test/OOS indices from 08_Split.R

Outputs:
    - features_latent.rds  : (permno, year, y, z1..z24, vae_recon_error)
    - vae_model/           : saved encoder + decoder weights
    - figures/vae_*.png    : training curves, latent space diagnostics

Design decisions:
    [1] PIT (uniform [0,1]) fitted on train set only, applied to test/OOS.
        Ensures no leakage of test distribution into training normalisation.
    [2] z_mean used for latent features (deterministic, stable).
        z_sampled used during training (reparameterisation trick).
    [3] Imputation: median per feature, fitted on train set only.
    [4] gamma=0 recovers standard beta-VAE. Set gamma>0 for supervised VAE.
    [5] Reconstruction error (per-row MSE) appended as anomaly feature.
    [6] recession column (binary) uses BCE reconstruction loss.
        All other features use MSE reconstruction loss.
"""

# ==============================================================================
# 0. Imports & Configuration
# ==============================================================================

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for server/PyCharm run
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Adjust ROOT to match your MT/ project root
ROOT          = Path(__file__).resolve().parents[2]   # MT/
DIR_DATA      = ROOT / "02_Data"
DIR_FEATURES  = DIR_DATA / "Features"
DIR_FIGURES   = ROOT / "03_Output" / "Figures"
DIR_MODELS    = ROOT / "03_Output" / "Models" / "VAE"

DIR_FIGURES.mkdir(parents=True, exist_ok=True)
DIR_MODELS.mkdir(parents=True,  exist_ok=True)

PATH_FEATURES_RAW    = DIR_FEATURES / "features_raw.rds"
PATH_SPLITS          = DIR_FEATURES / "splits.rds"
PATH_FEATURES_LATENT = DIR_FEATURES / "features_latent.rds"

# ── Hyperparameters ───────────────────────────────────────────────────────────
CFG = {
    # Architecture
    "z_dim"          : 24,           # Latent dimension: sqrt(463) ≈ 21 → 24
    "encoder_dims"   : [256, 128, 64],
    "decoder_dims"   : [64, 128, 256],
    "classifier_dims": [16],

    # Loss weights
    "beta"           : 3.0,          # KL weight — disentanglement pressure
    "gamma"          : 0.1,          # Classification loss weight (0 = pure VAE)

    # Training
    "epochs"         : 150,
    "batch_size"     : 512,
    "lr"             : 1e-3,
    "weight_decay"   : 1e-5,
    "patience"       : 15,           # Early stopping patience
    "kl_warmup"      : 20,           # Epochs to linearly ramp beta from 0

    # Reproducibility
    "seed"           : 42,
}

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[06C] Device: {DEVICE}")

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])


# ==============================================================================
# 1. Load Data
# ==============================================================================

print("[06C] Loading features_raw.rds...")
result       = pyreadr.read_r(str(PATH_FEATURES_RAW))
features_raw = result[None]   # pyreadr returns dict; None key = first object

print(f"[06C] features_raw: {features_raw.shape[0]:,} rows × {features_raw.shape[1]} cols")

# Load splits from R
print("[06C] Loading splits.rds...")
splits_result = pyreadr.read_r(str(PATH_SPLITS))
splits        = splits_result[None]

# splits expected to have columns: permno, year, split
# where split ∈ {"train", "test", "oos"}
assert "split" in splits.columns, "splits.rds must have a 'split' column"

# Merge split labels onto features
df = features_raw.merge(
    splits[["permno", "year", "split"]],
    on=["permno", "year"],
    how="left"
)

# Observations without a split label (e.g. pre-1998) → exclude
df = df[df["split"].notna()].reset_index(drop=True)
print(f"[06C] After split merge: {len(df):,} rows")
print(df["split"].value_counts().to_string())


# ==============================================================================
# 2. Feature Preparation
# ==============================================================================

# ── 2A. Identify column roles ─────────────────────────────────────────────────
ID_COLS   = ["permno", "year", "y", "censored", "param_id",
             "gvkey", "datadate", "lifetime_years", "fiscal_year_end_month", "split"]
BINARY_COLS = ["recession"]   # Only binary feature — BCE reconstruction loss

feature_cols = [c for c in df.columns
                if c not in ID_COLS
                and pd.api.types.is_numeric_dtype(df[c])]

# Separate continuous and binary feature indices (used for split reconstruction loss)
cont_cols   = [c for c in feature_cols if c not in BINARY_COLS]
bin_cols    = [c for c in feature_cols if c in BINARY_COLS]
col_order   = cont_cols + bin_cols   # continuous first, binary last
n_cont      = len(cont_cols)
n_binary    = len(bin_cols)
n_features  = len(col_order)

print(f"[06C] Features: {n_features} total | {n_cont} continuous | {n_binary} binary")

# ── 2B. Split ─────────────────────────────────────────────────────────────────
train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()
oos_df   = df[df["split"] == "oos"].copy()

print(f"[06C] Split sizes: train={len(train_df):,} | test={len(test_df):,} | oos={len(oos_df):,}")

X_train_raw = train_df[col_order].values.astype(np.float32)
X_test_raw  = test_df[col_order].values.astype(np.float32)
X_oos_raw   = oos_df[col_order].values.astype(np.float32)

y_train = train_df["y"].values.astype(np.float32)
y_test  = test_df["y"].values.astype(np.float32)

# ── 2C. Imputation (fitted on train only) ────────────────────────────────────
# Consecutive decline counters are count features — impute with 0, not median
consec_cols = [c for c in col_order if c.startswith("consec_decline_")]
other_cols  = [c for c in col_order if c not in consec_cols]

imputer_median = SimpleImputer(strategy="median")
imputer_zero   = SimpleImputer(strategy="constant", fill_value=0.0)

# Indices in col_order
consec_idx = [col_order.index(c) for c in consec_cols]
other_idx  = [col_order.index(c) for c in other_cols]

def impute(X_tr, X_te, X_oo):
    X_tr_out = X_tr.copy()
    X_te_out = X_te.copy()
    X_oo_out = X_oo.copy()

    if other_idx:
        imputer_median.fit(X_tr[:, other_idx])
        X_tr_out[:, other_idx] = imputer_median.transform(X_tr[:, other_idx])
        X_te_out[:, other_idx] = imputer_median.transform(X_te[:, other_idx])
        X_oo_out[:, other_idx] = imputer_median.transform(X_oo[:, other_idx])

    if consec_idx:
        imputer_zero.fit(X_tr[:, consec_idx])
        X_tr_out[:, consec_idx] = imputer_zero.transform(X_tr[:, consec_idx])
        X_te_out[:, consec_idx] = imputer_zero.transform(X_te[:, consec_idx])
        X_oo_out[:, consec_idx] = imputer_zero.transform(X_oo[:, consec_idx])

    return X_tr_out, X_te_out, X_oo_out

X_train_imp, X_test_imp, X_oos_imp = impute(X_train_raw, X_test_raw, X_oos_raw)

# Verify no NAs remain
assert not np.isnan(X_train_imp).any(), "NAs remain in train after imputation"
assert not np.isnan(X_test_imp).any(),  "NAs remain in test after imputation"
assert not np.isnan(X_oos_imp).any(),   "NAs remain in OOS after imputation"
print("[06C] Imputation complete — 0 NAs remaining")

# ── 2D. PIT normalisation: uniform [0,1] (fitted on train only) ──────────────
# Applied to continuous features only — binary recession stays as {0,1}
# n_output_distribution="uniform" maps to [0,1] range
pit = QuantileTransformer(
    output_distribution="uniform",
    n_quantiles=min(1000, len(X_train_imp)),
    random_state=CFG["seed"]
)

X_train_norm = X_train_imp.copy()
X_test_norm  = X_test_imp.copy()
X_oos_norm   = X_oos_imp.copy()

if n_cont > 0:
    pit.fit(X_train_imp[:, :n_cont])
    X_train_norm[:, :n_cont] = pit.transform(X_train_imp[:, :n_cont])
    X_test_norm[:, :n_cont]  = pit.transform(X_test_imp[:, :n_cont])
    X_oos_norm[:, :n_cont]   = pit.transform(X_oos_imp[:, :n_cont])

print(f"[06C] PIT applied to {n_cont} continuous features")
print(f"  Train range after PIT: [{X_train_norm[:, :n_cont].min():.4f}, "
      f"{X_train_norm[:, :n_cont].max():.4f}]")

# ── 2E. Tensors ───────────────────────────────────────────────────────────────
X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_norm,  dtype=torch.float32)
X_oos_t   = torch.tensor(X_oos_norm,   dtype=torch.float32)
y_train_t = torch.tensor(y_train,      dtype=torch.float32)
y_test_t  = torch.tensor(y_test,       dtype=torch.float32)

# Only include non-NA label rows in supervised loss
train_labelled_mask = ~torch.isnan(y_train_t)
test_labelled_mask  = ~torch.isnan(y_test_t)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=CFG["batch_size"],
    shuffle=True,
    drop_last=False
)

print(f"[06C] Train batches: {len(train_loader)}")


# ==============================================================================
# 3. Model Architecture
# ==============================================================================

class Encoder(nn.Module):
    """
    Maps input x → (z_mean, z_log_var).
    z_log_var is clamped to [-10, 10] for numerical stability.
    """
    def __init__(self, input_dim: int, hidden_dims: list, z_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        self.net     = nn.Sequential(*layers)
        self.mu      = nn.Linear(in_dim, z_dim)
        self.log_var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        h       = self.net(x)
        z_mu    = self.mu(h)
        z_lv    = self.log_var(h).clamp(-10, 10)
        return z_mu, z_lv


class Decoder(nn.Module):
    """
    Maps z → x_reconstructed.
    Continuous features: linear output (MSE loss).
    Binary features: sigmoid output (BCE loss).
    """
    def __init__(self, z_dim: int, hidden_dims: list,
                 n_cont: int, n_binary: int):
        super().__init__()
        layers = []
        in_dim = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        self.net      = nn.Sequential(*layers)
        self.out_cont = nn.Linear(in_dim, n_cont)   if n_cont   > 0 else None
        self.out_bin  = nn.Linear(in_dim, n_binary)  if n_binary > 0 else None

    def forward(self, z):
        h     = self.net(z)
        parts = []
        if self.out_cont is not None:
            parts.append(self.out_cont(h))          # Linear — MSE target
        if self.out_bin is not None:
            parts.append(torch.sigmoid(self.out_bin(h)))   # Sigmoid — BCE target
        return torch.cat(parts, dim=1)


class ClassifierHead(nn.Module):
    """
    Optional supervised head: z_mean → P(CSI).
    Plugged onto the encoder output during training.
    gamma=0 disables its gradient contribution.
    """
    def __init__(self, z_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        in_dim = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_mean):
        return torch.sigmoid(self.net(z_mean)).squeeze(-1)


class BetaVAE(nn.Module):
    """
    Beta-VAE with optional supervised classification loss.

    total_loss = MSE_recon + beta * KL + gamma * BCE_clf
    """
    def __init__(self, input_dim, encoder_dims, decoder_dims,
                 classifier_dims, z_dim, n_cont, n_binary):
        super().__init__()
        self.encoder    = Encoder(input_dim, encoder_dims, z_dim)
        self.decoder    = Decoder(z_dim, decoder_dims, n_cont, n_binary)
        self.classifier = ClassifierHead(z_dim, classifier_dims)
        self.n_cont     = n_cont
        self.n_binary   = n_binary
        self.z_dim      = z_dim

    def reparameterise(self, z_mu, z_lv):
        """Reparameterisation trick: z = mu + eps * std."""
        std = torch.exp(0.5 * z_lv)
        eps = torch.randn_like(std)
        return z_mu + eps * std

    def forward(self, x):
        z_mu, z_lv = self.encoder(x)
        z          = self.reparameterise(z_mu, z_lv)
        x_recon    = self.decoder(z)
        y_pred     = self.classifier(z_mu)   # Use z_mean — deterministic
        return x_recon, z_mu, z_lv, y_pred

    def compute_loss(self, x, x_recon, z_mu, z_lv, y_true,
                     beta, gamma, labelled_mask):
        """
        Mixed reconstruction loss:
          Continuous: MSE averaged across features
          Binary    : BCE averaged across features
        KL divergence: standard Gaussian prior
        Classification: BCE on labelled observations only
        """
        # ── Reconstruction loss ───────────────────────────────────────────────
        if self.n_cont > 0:
            mse = F.mse_loss(
                x_recon[:, :self.n_cont],
                x[:, :self.n_cont],
                reduction="mean"
            )
        else:
            mse = torch.tensor(0.0, device=x.device)

        if self.n_binary > 0:
            bce_recon = F.binary_cross_entropy(
                x_recon[:, self.n_cont:],
                x[:, self.n_cont:],
                reduction="mean"
            )
        else:
            bce_recon = torch.tensor(0.0, device=x.device)

        recon_loss = mse + bce_recon

        # ── KL divergence ─────────────────────────────────────────────────────
        # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_lv - z_mu.pow(2) - z_lv.exp(), dim=1)
        )

        # ── Supervised classification loss (labelled rows only) ───────────────
        if gamma > 0 and labelled_mask.sum() > 0:
            y_pred_lab = self.classifier(z_mu)[labelled_mask]
            y_true_lab = y_true[labelled_mask]
            clf_loss   = F.binary_cross_entropy(y_pred_lab, y_true_lab)
        else:
            clf_loss = torch.tensor(0.0, device=x.device)

        total = recon_loss + beta * kl_loss + gamma * clf_loss

        return {
            "total"     : total,
            "recon"     : recon_loss,
            "kl"        : kl_loss,
            "clf"       : clf_loss,
            "mse"       : mse,
            "bce_recon" : bce_recon,
        }


# ==============================================================================
# 4. Training
# ==============================================================================

def get_beta_schedule(epoch: int, max_beta: float, warmup_epochs: int) -> float:
    """Linear KL warmup from 0 to max_beta over warmup_epochs."""
    if warmup_epochs <= 0:
        return max_beta
    return min(max_beta, max_beta * (epoch + 1) / warmup_epochs)


def train_vae(model, loader, cfg, device):
    """
    Full training loop with:
      - KL warmup schedule
      - Early stopping (patience on total loss)
      - Best weight checkpointing
    """
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    history = {k: [] for k in ["total", "recon", "kl", "clf"]}

    best_loss   = float("inf")
    best_state  = None
    patience_ct = 0

    model.train()
    for epoch in range(cfg["epochs"]):

        beta_now = get_beta_schedule(epoch, cfg["beta"], cfg["kl_warmup"])
        epoch_losses = {k: 0.0 for k in history}
        n_batches = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            lab_mask = ~torch.isnan(y_batch)

            optimiser.zero_grad()

            x_recon, z_mu, z_lv, y_pred = model(x_batch)
            losses = model.compute_loss(
                x_batch, x_recon, z_mu, z_lv, y_batch,
                beta=beta_now,
                gamma=cfg["gamma"],
                labelled_mask=lab_mask
            )

            losses["total"].backward()
            # Gradient clipping — prevents exploding gradients with financial data
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        # Epoch averages
        for k in history:
            history[k].append(epoch_losses[k] / n_batches)

        scheduler.step(history["total"][-1])

        # Logging
        if epoch % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{cfg['epochs']} | "
                f"β={beta_now:.2f} | "
                f"total={history['total'][-1]:.4f} | "
                f"recon={history['recon'][-1]:.4f} | "
                f"kl={history['kl'][-1]:.4f} | "
                f"clf={history['clf'][-1]:.4f}"
            )

        # Early stopping
        if history["total"][-1] < best_loss - 1e-4:
            best_loss  = history["total"][-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1} (best: {best_loss:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Best weights restored (loss: {best_loss:.4f})")

    return history


# ==============================================================================
# 5. Encoding & Feature Extraction
# ==============================================================================

@torch.no_grad()
def encode(model, X_tensor, device, batch_size=1024):
    """
    Encode data → (z_mean, reconstruction_error_per_row).
    Uses z_mean — deterministic, more stable for downstream classification.
    """
    model.eval()
    z_means   = []
    recon_errs = []

    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
    for (x_batch,) in loader:
        x_batch   = x_batch.to(device)
        z_mu, z_lv = model.encoder(x_batch)
        z_samp     = model.reparameterise(z_mu, z_lv)
        x_recon    = model.decoder(z_samp)

        # Per-row MSE reconstruction error (anomaly feature)
        err = F.mse_loss(x_recon, x_batch, reduction="none").mean(dim=1)

        z_means.append(z_mu.cpu().numpy())
        recon_errs.append(err.cpu().numpy())

    z_means    = np.vstack(z_means)
    recon_errs = np.concatenate(recon_errs)
    return z_means, recon_errs


# ==============================================================================
# 6. Diagnostics
# ==============================================================================

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["total"],  label="Total",        linewidth=1.5)
    axes[0].plot(history["recon"],  label="Reconstruction", linewidth=1.5)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["kl"],  label="KL",           linewidth=1.5, color="orange")
    axes[1].plot(history["clf"], label="Classification", linewidth=1.5, color="red")
    axes[1].set_title("KL & Classification Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_latent_space(z_means, y_labels, save_path, n_dims=4):
    """Plot first n_dims latent dimensions coloured by CSI label."""
    valid_mask = ~np.isnan(y_labels)
    z_valid    = z_means[valid_mask]
    y_valid    = y_labels[valid_mask]

    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4))
    for i in range(n_dims):
        axes[i].scatter(
            z_valid[y_valid == 0, i],
            z_valid[y_valid == 0, min(i+1, z_means.shape[1]-1)],
            alpha=0.1, s=2, c="steelblue", label="Non-CSI"
        )
        axes[i].scatter(
            z_valid[y_valid == 1, i],
            z_valid[y_valid == 1, min(i+1, z_means.shape[1]-1)],
            alpha=0.3, s=4, c="crimson", label="CSI"
        )
        axes[i].set_title(f"z{i+1} vs z{i+2}")
        axes[i].set_xlabel(f"z{i+1}")
        axes[i].set_ylabel(f"z{i+2}")
    axes[0].legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def latent_diagnostics(z_train, y_train, z_test, recon_err_train, recon_err_test):
    """Print latent space quality metrics."""
    dim_var     = z_train.var(axis=0)
    n_collapsed = (dim_var < 1e-3).sum()

    print(f"\n  Latent space diagnostics:")
    print(f"    z_dim          : {z_train.shape[1]}")
    print(f"    Collapsed dims : {n_collapsed} (var < 1e-3)")
    print(f"    Dim var range  : [{dim_var.min():.4f}, {dim_var.max():.4f}]")

    # Reconstruction error by label
    valid = ~np.isnan(y_train)
    csi_err    = recon_err_train[valid & (y_train == 1)].mean()
    noncsi_err = recon_err_train[valid & (y_train == 0)].mean()
    print(f"    Recon err CSI     (train): {csi_err:.4f}")
    print(f"    Recon err Non-CSI (train): {noncsi_err:.4f}")
    print(f"    Ratio (CSI/Non-CSI)      : {csi_err/noncsi_err:.3f}  "
          f"(>1 = VAE learned anomaly signal)")
    print(f"    Recon err test p95       : {np.percentile(recon_err_test, 95):.4f}")


# ==============================================================================
# 7. Main
# ==============================================================================

def main():

    print(f"\n[06C] START — beta-VAE for CSI prediction")
    print(f"  Input features : {n_features}")
    print(f"  Continuous     : {n_cont}")
    print(f"  Binary         : {n_binary}")
    print(f"  z_dim          : {CFG['z_dim']}")
    print(f"  beta           : {CFG['beta']}")
    print(f"  gamma          : {CFG['gamma']}")

    # ── 7A. Build model ───────────────────────────────────────────────────────
    model = BetaVAE(
        input_dim       = n_features,
        encoder_dims    = CFG["encoder_dims"],
        decoder_dims    = CFG["decoder_dims"],
        classifier_dims = CFG["classifier_dims"],
        z_dim           = CFG["z_dim"],
        n_cont          = n_cont,
        n_binary        = n_binary
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # ── 7B. Train ─────────────────────────────────────────────────────────────
    print(f"\n[06C] Training...")
    history = train_vae(model, train_loader, CFG, DEVICE)

    # ── 7C. Save model weights ────────────────────────────────────────────────
    torch.save(model.state_dict(), DIR_MODELS / "vae_weights.pt")
    torch.save(model.encoder.state_dict(), DIR_MODELS / "encoder_weights.pt")
    with open(DIR_MODELS / "vae_config.json", "w") as f:
        json.dump({**CFG, "n_features": n_features,
                   "n_cont": n_cont, "n_binary": n_binary,
                   "col_order": col_order}, f, indent=2)
    print(f"  Weights saved to {DIR_MODELS}")

    # ── 7D. Encode all splits ─────────────────────────────────────────────────
    print("\n[06C] Encoding train / test / OOS...")
    z_train, err_train = encode(model, X_train_t, DEVICE)
    z_test,  err_test  = encode(model, X_test_t,  DEVICE)
    z_oos,   err_oos   = encode(model, X_oos_t,   DEVICE)

    # ── 7E. Diagnostics ───────────────────────────────────────────────────────
    latent_diagnostics(z_train, y_train, z_test, err_train, err_test)
    plot_training_curves(history, DIR_FIGURES / "vae_training_curves.png")
    plot_latent_space(z_train, y_train, DIR_FIGURES / "vae_latent_space.png")

    # ── 7F. Assemble features_latent ──────────────────────────────────────────
    z_cols = [f"z{i+1}" for i in range(CFG["z_dim"])]

    def build_latent_df(source_df, z_arr, err_arr):
        id_part = source_df[["permno", "year", "y", "censored"]].reset_index(drop=True)
        z_part  = pd.DataFrame(z_arr,  columns=z_cols)
        err_s   = pd.Series(err_arr, name="vae_recon_error")
        return pd.concat([id_part, z_part, err_s], axis=1)

    lat_train = build_latent_df(train_df, z_train, err_train)
    lat_test  = build_latent_df(test_df,  z_test,  err_test)
    lat_oos   = build_latent_df(oos_df,   z_oos,   err_oos)

    features_latent = pd.concat([lat_train, lat_test, lat_oos],
                                 ignore_index=True)

    print(f"\n[06C] features_latent: {features_latent.shape}")
    print(f"  Columns: {list(features_latent.columns)}")
    print(f"  NAs: {features_latent.isna().sum().sum()}")

    # ── 7G. Save to RDS via pyreadr (write as feather/parquet for R pickup) ───
    # pyreadr cannot write RDS directly from Python.
    # Save as parquet — read in R with arrow::read_parquet(PATH_FEATURES_LATENT)
    # Update PATH_FEATURES_LATENT in config.R to point to .parquet file.
    latent_path = DIR_FEATURES / "features_latent.parquet"
    features_latent.to_parquet(latent_path, index=False)
    print(f"  Saved: {latent_path}")
    print("  In R: arrow::read_parquet(file.path(DIR_FEATURES, 'features_latent.parquet'))")

    # ── 7H. Final assertions ──────────────────────────────────────────────────
    assert features_latent.isna().sum().sum() == 0, "NAs in features_latent"
    assert features_latent.shape[1] == len(z_cols) + 5, \
        f"Expected {len(z_cols)+5} cols, got {features_latent.shape[1]}"
    print("\n[06C] All assertions passed.")
    print(f"[06C] DONE")


if __name__ == "__main__":
    main()