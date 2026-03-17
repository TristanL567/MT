"""
08B_Autoencoder.py
==================
Beta-VAE with optional supervised classification loss for CSI prediction.

Architecture:
    Encoder     : n_features → 256 → 128 → 64 → (z_mean, z_log_var)
    Decoder     : z_dim → 64 → 128 → 256 → n_features
    Classifier  : z_mean → 16 → 1  (optional, controlled by gamma)

Loss:
    total = MSE_recon + beta * KL + gamma * BCE_classification

Inputs (from R pipeline):
    - features_raw.rds         : feature matrix (pyreadr)
    - split_labels_oot.parquet : OOT split labels (arrow parquet)

Outputs:
    - features_latent.parquet  : (permno, year, y, censored, z1..z24,
                                   vae_recon_error, split)
    - Models/VAE/              : encoder weights, full model weights,
                                 vae_config.json
    - Figures/                 : training curves, latent space plot

Design decisions:
    [1] Hardcoded DATA_ROOT — immune to PyCharm working directory issues.
        Update DATA_ROOT if you move the MT/ project folder.
    [2] Split labels loaded from parquet (split_labels_oot.parquet),
        exported by 08_Split.R via arrow::write_parquet().
        pyreadr cannot read nested R lists — parquet is the correct bridge.
    [3] PIT (uniform [0,1]) fitted on TRAIN set only, applied to test/OOS.
        Prevents test distribution from leaking into training normalisation.
    [4] Imputation fitted on TRAIN set only.
        consec_decline_* columns → zero imputation (count features).
        All other features → median imputation.
    [5] z_mean used for latent features (deterministic, stable).
        z_sampled used during training (reparameterisation trick).
    [6] gamma=0.0 → pure beta-VAE (unsupervised).
        gamma>0.0 → supervised beta-VAE (latent space nudged toward CSI).
        Run both and compare Average Precision — addresses thesis subQ1.
    [7] recession column (binary {0,1}) → BCE reconstruction loss.
        All other features → MSE reconstruction loss.
    [8] LayerNorm instead of BatchNorm — normalises within each observation,
        not across the batch. Correct for high-variance financial data.
    [9] GELU activation — smoother than ReLU, better for near-zero financial
        ratios, standard in modern tabular models.
"""

# ==============================================================================
# 0. Imports
# ==============================================================================

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
matplotlib.use("Agg")   # Non-interactive — safe for PyCharm run configs
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. Paths  — update DATA_ROOT if MT/ project moves
# ==============================================================================

DATA_ROOT    = Path(r"C:\Users\Tristan Leiter\Documents\MT")
DIR_DATA     = DATA_ROOT / "02_Data"
DIR_FEATURES = DIR_DATA  / "Features"
DIR_FIGURES  = DATA_ROOT / "03_Output" / "Figures"
DIR_MODELS   = DATA_ROOT / "03_Output" / "Models" / "VAE"

DIR_FIGURES.mkdir(parents=True, exist_ok=True)
DIR_MODELS.mkdir(parents=True,  exist_ok=True)

PATH_FEATURES_RAW    = DIR_FEATURES / "features_raw.rds"
PATH_SPLIT_LABELS    = DIR_FEATURES / "split_labels_oot.parquet"
PATH_FEATURES_LATENT = DIR_FEATURES / "features_latent.parquet"

# Verify critical inputs exist before doing anything else
for p in [PATH_FEATURES_RAW, PATH_SPLIT_LABELS]:
    assert p.exists(), f"Required input not found: {p}\nRun 08_Split.R first."

print(f"[08B] DATA_ROOT    : {DATA_ROOT}")
print(f"[08B] DIR_FEATURES : {DIR_FEATURES}")
print(f"[08B] Inputs verified.")

# ==============================================================================
# 2. Hyperparameters
# ==============================================================================

CFG = {
    # Architecture
    "z_dim"          : 24,            # Latent dim — sqrt(463) ≈ 21 → 24
    "encoder_dims"   : [256, 128, 64],
    "decoder_dims"   : [64, 128, 256],
    "classifier_dims": [16],

    # Loss weights
    "beta"           : 3.0,           # KL weight — disentanglement pressure
    "gamma"          : 0.1,           # Classification loss weight (0 = pure VAE)

    # Training
    "epochs"         : 150,
    "batch_size"     : 512,
    "lr"             : 1e-3,
    "weight_decay"   : 1e-5,
    "patience"       : 15,            # Early stopping patience (epochs)
    "kl_warmup"      : 20,            # Epochs to linearly ramp beta from 0→beta

    # Reproducibility
    "seed"           : 42,
}

# ==============================================================================
# 3. Device & Seeds
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[08B] Device: {DEVICE}")

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ==============================================================================
# 4. Load Data
# ==============================================================================

print("\n[08B] Loading features_raw.rds...")
result       = pyreadr.read_r(str(PATH_FEATURES_RAW))
features_raw = result[None]   # pyreadr: None key = first object
print(f"  Shape: {features_raw.shape[0]:,} rows × {features_raw.shape[1]} cols")

print("[08B] Loading split labels (OOT)...")
split_labels = pd.read_parquet(PATH_SPLIT_LABELS)
print(f"  Split distribution:\n{split_labels['split'].value_counts().to_string()}")

# Merge split labels onto feature matrix
df = features_raw.merge(split_labels, on=["permno", "year"], how="left")

# Drop rows with no split assignment (e.g. pre-START_DATE observations)
n_before = len(df)
df = df[df["split"].notna()].reset_index(drop=True)
n_dropped = n_before - len(df)
if n_dropped > 0:
    print(f"  Dropped {n_dropped:,} rows with no split label")
print(f"  Rows after merge: {len(df):,}")

# ==============================================================================
# 5. Feature Preparation
# ==============================================================================

# ── 5A. Identify columns ──────────────────────────────────────────────────────
ID_COLS = [
    "permno", "year", "y", "censored", "param_id",
    "gvkey", "datadate", "lifetime_years", "fiscal_year_end_month", "split"
]

# Binary features — BCE reconstruction loss
BINARY_COLS = ["recession"]

# All numeric non-ID columns are features
feature_cols = [
    c for c in df.columns
    if c not in ID_COLS
    and pd.api.types.is_numeric_dtype(df[c])
]

# Ordered: continuous first, binary last (required for split reconstruction loss)
cont_cols  = [c for c in feature_cols if c not in BINARY_COLS]
bin_cols   = [c for c in feature_cols if c in BINARY_COLS]
col_order  = cont_cols + bin_cols
n_cont     = len(cont_cols)
n_binary   = len(bin_cols)
n_features = len(col_order)

print(f"\n[08B] Feature classification:")
print(f"  Total features : {n_features}")
print(f"  Continuous     : {n_cont}")
print(f"  Binary         : {n_binary}  {bin_cols}")

# ── 5B. Split into train / test / OOS ─────────────────────────────────────────
train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
test_df  = df[df["split"] == "test"].copy().reset_index(drop=True)
oos_df   = df[df["split"] == "oos"].copy().reset_index(drop=True)

print(f"\n[08B] Split sizes:")
print(f"  Train : {len(train_df):,} rows")
print(f"  Test  : {len(test_df):,} rows")
print(f"  OOS   : {len(oos_df):,} rows")

X_train_raw = train_df[col_order].values.astype(np.float32)
X_test_raw  = test_df[col_order].values.astype(np.float32)
X_oos_raw   = oos_df[col_order].values.astype(np.float32)

y_train = train_df["y"].values.astype(np.float32)
y_test  = test_df["y"].values.astype(np.float32)

# ── 5C. Imputation — fitted on train only ─────────────────────────────────────
# consec_decline_* are count features (0, 1, 2, ...) — impute with 0
# All other features — impute with train-set median
consec_cols = [c for c in col_order if c.startswith("consec_decline_")]
other_cols  = [c for c in col_order if c not in consec_cols]

consec_idx = [col_order.index(c) for c in consec_cols]
other_idx  = [col_order.index(c) for c in other_cols]

imputer_median = SimpleImputer(strategy="median")
imputer_zero   = SimpleImputer(strategy="constant", fill_value=0.0)

def impute_splits(X_tr, X_te, X_oo):
    """Fit imputers on train only, apply to all splits."""
    out_tr = X_tr.copy()
    out_te = X_te.copy()
    out_oo = X_oo.copy()

    if other_idx:
        imputer_median.fit(X_tr[:, other_idx])
        out_tr[:, other_idx] = imputer_median.transform(X_tr[:, other_idx])
        out_te[:, other_idx] = imputer_median.transform(X_te[:, other_idx])
        out_oo[:, other_idx] = imputer_median.transform(X_oo[:, other_idx])

    if consec_idx:
        imputer_zero.fit(X_tr[:, consec_idx])
        out_tr[:, consec_idx] = imputer_zero.transform(X_tr[:, consec_idx])
        out_te[:, consec_idx] = imputer_zero.transform(X_te[:, consec_idx])
        out_oo[:, consec_idx] = imputer_zero.transform(X_oo[:, consec_idx])

    return out_tr, out_te, out_oo

print("\n[08B] Imputing missing values (train-fit only)...")
X_train_imp, X_test_imp, X_oos_imp = impute_splits(
    X_train_raw, X_test_raw, X_oos_raw
)

assert not np.isnan(X_train_imp).any(), "NAs remain in train after imputation"
assert not np.isnan(X_test_imp).any(),  "NAs remain in test after imputation"
assert not np.isnan(X_oos_imp).any(),   "NAs remain in OOS after imputation"
print("  Imputation complete — 0 NAs remaining in all splits")

# ── 5D. PIT normalisation — uniform [0,1] fitted on train only ───────────────
# Applied to continuous features only.
# Binary recession column stays as {0,1} — PIT would destroy its meaning.
print("\n[08B] Applying PIT normalisation (train-fit only)...")

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

print(f"  PIT applied to {n_cont} continuous features")
print(f"  Train range after PIT: "
      f"[{X_train_norm[:, :n_cont].min():.4f}, "
      f"{X_train_norm[:, :n_cont].max():.4f}]")

# ── 5E. Build PyTorch tensors and DataLoader ──────────────────────────────────
X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_norm,  dtype=torch.float32)
X_oos_t   = torch.tensor(X_oos_norm,   dtype=torch.float32)
y_train_t = torch.tensor(y_train,      dtype=torch.float32)
y_test_t  = torch.tensor(y_test,       dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=CFG["batch_size"],
    shuffle=True,
    drop_last=False
)

print(f"\n[08B] DataLoader: {len(train_loader)} batches per epoch "
      f"(batch_size={CFG['batch_size']})")

# ==============================================================================
# 6. Model Architecture
# ==============================================================================

class Encoder(nn.Module):
    """
    x → (z_mean, z_log_var)
    LayerNorm + GELU: better than BatchNorm+ReLU for financial tabular data.
    z_log_var clamped to [-10, 10] for numerical stability.
    """
    def __init__(self, input_dim: int, hidden_dims: list, z_dim: int):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        self.net     = nn.Sequential(*layers)
        self.mu      = nn.Linear(in_d, z_dim)
        self.log_var = nn.Linear(in_d, z_dim)

    def forward(self, x):
        h    = self.net(x)
        z_mu = self.mu(h)
        z_lv = self.log_var(h).clamp(-10, 10)
        return z_mu, z_lv


class Decoder(nn.Module):
    """
    z → x_reconstructed
    Split output heads:
      out_cont : linear activation → MSE loss for continuous features
      out_bin  : sigmoid activation → BCE loss for binary features
    """
    def __init__(self, z_dim: int, hidden_dims: list,
                 n_cont: int, n_binary: int):
        super().__init__()
        layers = []
        in_d = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        self.net      = nn.Sequential(*layers)
        self.out_cont = nn.Linear(in_d, n_cont)   if n_cont   > 0 else None
        self.out_bin  = nn.Linear(in_d, n_binary)  if n_binary > 0 else None

    def forward(self, z):
        h     = self.net(z)
        parts = []
        if self.out_cont is not None:
            parts.append(self.out_cont(h))
        if self.out_bin is not None:
            parts.append(torch.sigmoid(self.out_bin(h)))
        return torch.cat(parts, dim=1)


class ClassifierHead(nn.Module):
    """
    z_mean → P(CSI)
    Only contributes to the loss when gamma > 0.
    Uses z_mean (deterministic) not z_sampled — more stable for classification.
    """
    def __init__(self, z_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        in_d = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.GELU()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_mean):
        return torch.sigmoid(self.net(z_mean)).squeeze(-1)


class BetaVAE(nn.Module):
    """
    Beta-VAE with optional supervised classification head.

    total_loss = MSE_recon        (continuous reconstruction)
               + BCE_recon        (binary reconstruction)
               + beta  * KL       (disentanglement)
               + gamma * BCE_clf  (CSI discrimination, gamma=0 → pure VAE)
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
        """z = mu + eps * std,  eps ~ N(0, I)"""
        std = torch.exp(0.5 * z_lv)
        eps = torch.randn_like(std)
        return z_mu + eps * std

    def forward(self, x):
        z_mu, z_lv = self.encoder(x)
        z          = self.reparameterise(z_mu, z_lv)
        x_recon    = self.decoder(z)
        y_pred     = self.classifier(z_mu)
        return x_recon, z_mu, z_lv, y_pred

    def compute_loss(self, x, x_recon, z_mu, z_lv,
                     y_true, beta, gamma, labelled_mask):

        # Continuous reconstruction — MSE
        if self.n_cont > 0:
            mse = F.mse_loss(
                x_recon[:, :self.n_cont],
                x[:, :self.n_cont],
                reduction="mean"
            )
        else:
            mse = torch.tensor(0.0, device=x.device)

        # Binary reconstruction — BCE
        if self.n_binary > 0:
            bce_recon = F.binary_cross_entropy(
                x_recon[:, self.n_cont:],
                x[:, self.n_cont:],
                reduction="mean"
            )
        else:
            bce_recon = torch.tensor(0.0, device=x.device)

        recon_loss = mse + bce_recon

        # KL divergence — standard Gaussian prior
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_lv - z_mu.pow(2) - z_lv.exp(), dim=1)
        )

        # Supervised classification loss — labelled rows only
        # y=NA rows are excluded via labelled_mask
        if gamma > 0 and labelled_mask.sum() > 0:
            y_pred_lab = self.classifier(z_mu)[labelled_mask]
            y_true_lab = y_true[labelled_mask]
            clf_loss   = F.binary_cross_entropy(y_pred_lab, y_true_lab)
        else:
            clf_loss = torch.tensor(0.0, device=x.device)

        total = recon_loss + beta * kl_loss + gamma * clf_loss

        return {
            "total"    : total,
            "recon"    : recon_loss,
            "kl"       : kl_loss,
            "clf"      : clf_loss,
            "mse"      : mse,
            "bce_recon": bce_recon,
        }


# ==============================================================================
# 7. Training
# ==============================================================================

def get_beta(epoch: int, max_beta: float, warmup: int) -> float:
    """Linear KL warmup: beta ramps from 0 → max_beta over warmup epochs."""
    if warmup <= 0:
        return max_beta
    return min(max_beta, max_beta * (epoch + 1) / warmup)


def train_vae(model, loader, cfg, device):
    """
    Training loop:
      - Adam optimiser with weight decay
      - ReduceLROnPlateau scheduler
      - KL warmup schedule
      - Gradient clipping (max_norm=1.0)
      - Early stopping with best-weight checkpointing
    """
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    history     = {k: [] for k in ["total", "recon", "kl", "clf"]}
    best_loss   = float("inf")
    best_state  = None
    patience_ct = 0

    model.train()

    for epoch in range(cfg["epochs"]):

        beta_now     = get_beta(epoch, cfg["beta"], cfg["kl_warmup"])
        epoch_losses = {k: 0.0 for k in history}
        n_batches    = 0

        for x_batch, y_batch in loader:
            x_batch  = x_batch.to(device)
            y_batch  = y_batch.to(device)
            lab_mask = ~torch.isnan(y_batch)   # Exclude y=NA from clf loss

            optimiser.zero_grad()

            x_recon, z_mu, z_lv, y_pred = model(x_batch)
            losses = model.compute_loss(
                x_batch, x_recon, z_mu, z_lv, y_batch,
                beta=beta_now, gamma=cfg["gamma"],
                labelled_mask=lab_mask
            )

            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        # Record epoch averages
        for k in history:
            history[k].append(epoch_losses[k] / n_batches)

        scheduler.step(history["total"][-1])

        # Log every 10 epochs
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
            best_loss   = history["total"][-1]
            best_state  = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best loss: {best_loss:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Best weights restored (loss: {best_loss:.4f})")

    return history


# ==============================================================================
# 8. Encoding
# ==============================================================================

@torch.no_grad()
def encode(model, X_tensor, device, batch_size=1024):
    """
    Encode full dataset → (z_mean array, per-row reconstruction error).
    z_mean is deterministic — preferred over z_sampled for downstream models.
    Reconstruction error = per-row MSE → anomaly feature.
    """
    model.eval()
    z_list   = []
    err_list = []

    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
    for (x_batch,) in loader:
        x_batch    = x_batch.to(device)
        z_mu, z_lv = model.encoder(x_batch)
        z_samp     = model.reparameterise(z_mu, z_lv)
        x_recon    = model.decoder(z_samp)
        err        = F.mse_loss(x_recon, x_batch, reduction="none").mean(dim=1)

        z_list.append(z_mu.cpu().numpy())
        err_list.append(err.cpu().numpy())

    return np.vstack(z_list), np.concatenate(err_list)


# ==============================================================================
# 9. Diagnostics & Plots
# ==============================================================================

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["total"], label="Total",          linewidth=1.5)
    axes[0].plot(history["recon"], label="Reconstruction", linewidth=1.5)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["kl"],  label="KL Divergence",    linewidth=1.5,
                 color="orange")
    axes[1].plot(history["clf"], label="Classification",   linewidth=1.5,
                 color="red")
    axes[1].set_title("KL & Classification Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_latent_space(z_means, y_labels, save_path, n_pairs=4):
    """Scatter plot of first n_pairs latent dimensions, coloured by CSI."""
    valid   = ~np.isnan(y_labels)
    z_v     = z_means[valid]
    y_v     = y_labels[valid]
    n_dims  = z_means.shape[1]

    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4))
    for i in range(n_pairs):
        j = min(i + 1, n_dims - 1)
        axes[i].scatter(z_v[y_v == 0, i], z_v[y_v == 0, j],
                        alpha=0.1, s=2, c="steelblue", label="Non-CSI")
        axes[i].scatter(z_v[y_v == 1, i], z_v[y_v == 1, j],
                        alpha=0.3, s=4, c="crimson",   label="CSI")
        axes[i].set_title(f"z{i+1} vs z{j+1}")
        axes[i].set_xlabel(f"z{i+1}")
        axes[i].set_ylabel(f"z{j+1}")
    axes[0].legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def print_latent_diagnostics(z_train, y_train, z_test,
                              err_train, err_test):
    dim_var     = z_train.var(axis=0)
    n_collapsed = int((dim_var < 1e-3).sum())

    print(f"\n  Latent space diagnostics:")
    print(f"    z_dim              : {z_train.shape[1]}")
    print(f"    Collapsed dims     : {n_collapsed}  "
          f"(var < 1e-3 — increase β if > 0)")
    print(f"    Dim variance range : [{dim_var.min():.4f}, "
          f"{dim_var.max():.4f}]")

    valid      = ~np.isnan(y_train)
    csi_err    = err_train[valid & (y_train == 1)].mean()
    noncsi_err = err_train[valid & (y_train == 0)].mean()
    ratio      = csi_err / noncsi_err if noncsi_err > 0 else float("nan")

    print(f"    Recon err CSI      : {csi_err:.4f}")
    print(f"    Recon err Non-CSI  : {noncsi_err:.4f}")
    print(f"    Ratio (CSI/Non-CSI): {ratio:.3f}  "
          f"(>1 means VAE learned anomaly signal)")
    print(f"    Test recon p95     : {np.percentile(err_test, 95):.4f}")


# ==============================================================================
# 10. Main
# ==============================================================================

def main():

    print(f"\n{'='*60}")
    print(f"[08B] Beta-VAE for CSI prediction — START")
    print(f"{'='*60}")
    print(f"  Input features : {n_features}")
    print(f"  Continuous     : {n_cont}")
    print(f"  Binary         : {n_binary}  {bin_cols}")
    print(f"  z_dim          : {CFG['z_dim']}")
    print(f"  beta           : {CFG['beta']}")
    print(f"  gamma          : {CFG['gamma']}  "
          f"({'supervised' if CFG['gamma'] > 0 else 'pure VAE'})")

    # ── 10A. Build model ──────────────────────────────────────────────────────
    model = BetaVAE(
        input_dim       = n_features,
        encoder_dims    = CFG["encoder_dims"],
        decoder_dims    = CFG["decoder_dims"],
        classifier_dims = CFG["classifier_dims"],
        z_dim           = CFG["z_dim"],
        n_cont          = n_cont,
        n_binary        = n_binary,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[08B] Model parameters: {n_params:,}")
    print(f"  Encoder: {n_features} → "
          f"{' → '.join(str(d) for d in CFG['encoder_dims'])} "
          f"→ z({CFG['z_dim']})")
    print(f"  Decoder: z({CFG['z_dim']}) → "
          f"{' → '.join(str(d) for d in CFG['decoder_dims'])} "
          f"→ {n_features}")

    # ── 10B. Train ────────────────────────────────────────────────────────────
    print(f"\n[08B] Training...")
    history = train_vae(model, train_loader, CFG, DEVICE)

    # ── 10C. Save model ───────────────────────────────────────────────────────
    torch.save(model.state_dict(),
               DIR_MODELS / "vae_weights.pt")
    torch.save(model.encoder.state_dict(),
               DIR_MODELS / "encoder_weights.pt")

    cfg_save = {
        **CFG,
        "n_features" : n_features,
        "n_cont"     : n_cont,
        "n_binary"   : n_binary,
        "col_order"  : col_order,
        "binary_cols": bin_cols,
    }
    with open(DIR_MODELS / "vae_config.json", "w") as f:
        json.dump(cfg_save, f, indent=2)

    print(f"\n[08B] Model saved to: {DIR_MODELS}")

    # ── 10D. Encode all splits ────────────────────────────────────────────────
    print("\n[08B] Encoding train / test / OOS...")
    z_train, err_train = encode(model, X_train_t, DEVICE)
    z_test,  err_test  = encode(model, X_test_t,  DEVICE)
    z_oos,   err_oos   = encode(model, X_oos_t,   DEVICE)

    # ── 10E. Diagnostics ──────────────────────────────────────────────────────
    print_latent_diagnostics(z_train, y_train, z_test, err_train, err_test)
    plot_training_curves(history,
                         DIR_FIGURES / "vae_training_curves.png")
    plot_latent_space(z_train, y_train,
                      DIR_FIGURES / "vae_latent_space_train.png")

    # ── 10F. Assemble features_latent ─────────────────────────────────────────
    z_cols = [f"z{i+1}" for i in range(CFG["z_dim"])]

    def build_latent_df(src_df, z_arr, err_arr, split_name):
        id_cols_keep = ["permno", "year", "y", "censored"]
        id_part = src_df[id_cols_keep].reset_index(drop=True)
        z_part  = pd.DataFrame(z_arr, columns=z_cols)
        err_col = pd.Series(err_arr, name="vae_recon_error")
        spl_col = pd.Series([split_name] * len(src_df), name="split")
        return pd.concat([id_part, z_part, err_col, spl_col], axis=1)

    lat_train = build_latent_df(train_df, z_train, err_train, "train")
    lat_test  = build_latent_df(test_df,  z_test,  err_test,  "test")
    lat_oos   = build_latent_df(oos_df,   z_oos,   err_oos,   "oos")

    features_latent = pd.concat(
        [lat_train, lat_test, lat_oos], ignore_index=True
    )

    print(f"\n[08B] features_latent shape: {features_latent.shape}")
    print(f"  Columns: {list(features_latent.columns)}")
    print(f"  NAs    : {features_latent.isna().sum().sum()}")

    # ── 10G. Save as parquet (readable in R via arrow::read_parquet) ──────────
    features_latent.to_parquet(PATH_FEATURES_LATENT, index=False)
    print(f"\n[08B] Saved: {PATH_FEATURES_LATENT}")
    print(f"  Load in R: arrow::read_parquet(PATH_FEATURES_LATENT)")

    # ── 10H. Final assertions ─────────────────────────────────────────────────
    assert features_latent.isna().sum().sum() == 0, \
        "NAs found in features_latent — check encoding step"
    assert "vae_recon_error" in features_latent.columns, \
        "vae_recon_error missing from features_latent"
    assert all(c in features_latent.columns for c in z_cols), \
        "Latent dimension columns missing"
    assert features_latent["split"].nunique() == 3, \
        "Expected 3 split values (train/test/oos)"

    print("\n[08B] All assertions passed.")
    print(f"[08B] DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()