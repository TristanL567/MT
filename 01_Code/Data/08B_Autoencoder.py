"""
08B_Autoencoder.py  (v3 — dual input mode)
==========================================
Beta-VAE with optional supervised classification loss for CSI prediction.

VAE INPUT SELECTION:
    Set VAE_INPUT at the top of Section 3 (Configuration):

    VAE_INPUT = "fund"  → M2: VAE trained on fundamentals only
                              Input : features_fund.rds
                              Output: features_latent_fund.parquet
                              Purpose: Does VAE add signal over raw fundamentals?

    VAE_INPUT = "raw"   → M4: VAE trained on full feature set
                              Input : features_raw.rds
                              Output: features_latent_raw.parquet
                              Purpose: Does VAE compress full features usefully?

    Run twice — once per VAE_INPUT — to produce both latent files.

Architecture:
    Encoder     : n_features → 256 → 128 → 64 → (z_mean, z_log_var)
    Decoder     : z_dim → 64 → 128 → 256 → n_features
    Classifier  : z_mean → 16 → 1  (logits, optional, controlled by gamma)

Loss (corrected ELBO):
    total = recon_loss + beta * KL + gamma * BCE_clf

    recon_loss = MSE.sum(dim=1).mean()     ← FIX 1: sum over features first
               + BCE.sum(dim=1).mean()
    KL         = -0.5 * mean(sum(1 + lv - mu^2 - exp(lv)))
    BCE_clf    = BCEWithLogitsLoss         ← FIX 4: numerically stable

    Both recon_loss and KL now average over the batch dimension only,
    making them comparable in scale. This is the correct ELBO formulation.

PIT normalisation:
    output_distribution="normal"           ← FIX 2: matches linear decoder
    Fitted on TRAIN set only.

Early stopping:
    Validation split (15% of train)        ← FIX 3: generalisation check
    Early stopping on VALIDATION loss.

Outputs:
    VAE_INPUT="fund" → features_latent_fund.parquet  [M2]
    VAE_INPUT="raw"  → features_latent_raw.parquet   [M4]
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
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. Utility functions
# ==============================================================================

def clip_to_float32_safe(X: np.ndarray,
                         quantile_low:  float = 0.001,
                         quantile_high: float = 0.999) -> np.ndarray:
    X = X.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    lo = np.nanpercentile(X, quantile_low  * 100, axis=0)
    hi = np.nanpercentile(X, quantile_high * 100, axis=0)
    X  = np.clip(X, lo, hi)
    return X.astype(np.float32)


def get_valid_cols(df_tr, df_te, df_oo, cols):
    valid   = []
    dropped = []
    for c in cols:
        if (df_tr[c].notna().any() and
                df_te[c].notna().any() and
                df_oo[c].notna().any()):
            valid.append(c)
        else:
            dropped.append(c)
    if dropped:
        print(f"  Dropped {len(dropped)} all-NaN cols: "
              f"{sorted(dropped)[:5]}"
              f"{'...' if len(dropped) > 5 else ''}")
    return valid


# ==============================================================================
# 2. Paths
# ==============================================================================

if os.name == "nt":
    DATA_ROOT = Path(r"C:\Users\Tristan Leiter\Documents\MT")
else:
    DATA_ROOT = Path("/workspace/MT")   # standard Vast.ai working directory

DIR_DATA     = DATA_ROOT / "02_Data"
DIR_FEATURES = DIR_DATA  / "Features"
DIR_LABELS   = DIR_DATA  / "Labels"
DIR_OUTPUT   = DATA_ROOT / "03_Output"
DIR_MODELS   = DIR_OUTPUT / "Models" / "AutoGluon"
DIR_TABLES   = DIR_OUTPUT / "Tables"

DIR_FIGURES.mkdir(parents=True, exist_ok=True)
DIR_MODELS.mkdir(parents=True,  exist_ok=True)

PATH_SPLIT_LABELS = DIR_FEATURES / "split_labels_oot.parquet"
assert PATH_SPLIT_LABELS.exists(), (
    f"Required input not found: {PATH_SPLIT_LABELS}\n"
    f"Run 08_Split.R first."
)

# ==============================================================================
# 3. ── VAE INPUT SELECTION ────────────────────────────────────────────────────
#
#   VAE_INPUT = "fund"  → M2: VAE on fundamentals only (no price features)
#                             features_fund.rds → features_latent_fund.parquet
#
#   VAE_INPUT = "raw"   → M4: VAE on full feature set
#                             features_raw.rds  → features_latent_raw.parquet
#
# ==============================================================================

VAE_INPUT = "M2"    # ← CHANGE THIS: "fund" | "raw"

VAE_INPUT_CONFIG = {
    "fund": {
        "path"        : DIR_FEATURES / "features_fund.rds",
        "out"         : DIR_FEATURES / "features_latent_fund.parquet",
        "model_dir"   : DIR_MODELS / "fund",
        "fig_suffix"  : "fund",
        "description" : "M2 — VAE on fundamentals only (no price features)",
    },
    "raw": {
        "path"        : DIR_FEATURES / "features_raw.rds",
        "out"         : DIR_FEATURES / "features_latent_raw.parquet",
        "model_dir"   : DIR_MODELS / "raw",
        "fig_suffix"  : "raw",
        "description" : "M4 — VAE on full feature set",
    },
}

assert VAE_INPUT in VAE_INPUT_CONFIG, \
    f"Unknown VAE_INPUT '{VAE_INPUT}'. Choose from: {list(VAE_INPUT_CONFIG.keys())}"

vcfg = VAE_INPUT_CONFIG[VAE_INPUT]
vcfg["model_dir"].mkdir(parents=True, exist_ok=True)

PATH_FEATURES_INPUT  = vcfg["path"]
PATH_FEATURES_LATENT = vcfg["out"]
DIR_MODELS_RUN       = vcfg["model_dir"]

assert PATH_FEATURES_INPUT.exists(), \
    f"Feature file not found: {PATH_FEATURES_INPUT}\n" \
    f"Run 06B_Feature_Eng.R first."

print(f"[08B] ══════════════════════════════════════")
print(f"  VAE_INPUT    : {VAE_INPUT.upper()}")
print(f"  Description  : {vcfg['description']}")
print(f"  Input file   : {PATH_FEATURES_INPUT.name}")
print(f"  Output file  : {PATH_FEATURES_LATENT.name}")
print(f"  Model dir    : {DIR_MODELS_RUN}")
print(f"[08B] ══════════════════════════════════════\n")

# ==============================================================================
# 4. Hyperparameters
# ==============================================================================

CFG = {
    # Architecture
    "z_dim"          : 24,
    "encoder_dims"   : [256, 128, 64],
    "decoder_dims"   : [64, 128, 256],
    "classifier_dims": [16],

    # Loss weights
    "beta"           : 1.0,
    "gamma"          : 0.1,

    # Training
    "epochs"         : 150,
    "batch_size"     : 512,
    "lr"             : 1e-3,
    "weight_decay"   : 1e-5,
    "patience"       : 15,
    "kl_warmup"      : 20,
    "val_fraction"   : 0.15,

    # Reproducibility
    "seed"           : 123,
}

# ==============================================================================
# 5. Device & Seeds
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[08B] Device: {DEVICE}")

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ==============================================================================
# 6. Load Data
# ==============================================================================

print(f"\n[08B] Loading {PATH_FEATURES_INPUT.name}...")
result        = pyreadr.read_r(str(PATH_FEATURES_INPUT))
features_input = result[None]
print(f"  Shape: {features_input.shape[0]:,} rows × {features_input.shape[1]} cols")

print("[08B] Loading split labels (OOT)...")
split_labels = pd.read_parquet(PATH_SPLIT_LABELS)
print(f"  Split distribution:\n"
      f"{split_labels['split'].value_counts().to_string()}")

df        = features_input.merge(split_labels, on=["permno", "year"], how="left")
n_before  = len(df)
df        = df[df["split"].notna()].reset_index(drop=True)
n_dropped = n_before - len(df)
if n_dropped > 0:
    print(f"  Dropped {n_dropped:,} rows with no split label")
print(f"  Rows after merge: {len(df):,}")

# ==============================================================================
# 7. Feature Preparation
# ==============================================================================

ID_COLS = [
    "permno", "year", "y", "censored", "param_id",
    "gvkey", "datadate", "lifetime_years", "fiscal_year_end_month", "split"
]
BINARY_COLS = ["recession"]

feature_cols = [
    c for c in df.columns
    if c not in ID_COLS
    and pd.api.types.is_numeric_dtype(df[c])
]

cont_cols_raw = [c for c in feature_cols if c not in BINARY_COLS]
bin_cols_raw  = [c for c in feature_cols if c in BINARY_COLS]

train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
test_df  = df[df["split"] == "test"].copy().reset_index(drop=True)
oos_df   = df[df["split"] == "oos"].copy().reset_index(drop=True)

print(f"\n[08B] OOT split sizes:")
print(f"  Train : {len(train_df):,}")
print(f"  Test  : {len(test_df):,}")
print(f"  OOS   : {len(oos_df):,}")

cont_cols  = get_valid_cols(train_df, test_df, oos_df, cont_cols_raw)
bin_cols   = get_valid_cols(train_df, test_df, oos_df, bin_cols_raw)
col_order  = cont_cols + bin_cols
n_cont     = len(cont_cols)
n_binary   = len(bin_cols)
n_features = len(col_order)

print(f"\n[08B] Feature classification:")
print(f"  Total     : {n_features}")
print(f"  Continuous: {n_cont}")
print(f"  Binary    : {n_binary}  {bin_cols}")

X_train_raw = clip_to_float32_safe(train_df[col_order].values.astype(np.float64))
X_test_raw  = clip_to_float32_safe(test_df[col_order].values.astype(np.float64))
X_oos_raw   = clip_to_float32_safe(oos_df[col_order].values.astype(np.float64))

y_train_full = train_df["y"].values.astype(np.float32)
y_test       = test_df["y"].values.astype(np.float32)

# Imputation
consec_cols  = [c for c in col_order if c.startswith("consec_decline_")]
consec_mask  = np.array([c in consec_cols for c in col_order])
other_mask   = ~consec_mask

imputer_median = SimpleImputer(strategy="median")
imputer_zero   = SimpleImputer(strategy="constant", fill_value=0.0)

def impute_splits(X_tr, X_te, X_oo):
    out_tr, out_te, out_oo = X_tr.copy(), X_te.copy(), X_oo.copy()
    if other_mask.any():
        imputer_median.fit(X_tr[:, other_mask])
        out_tr[:, other_mask] = imputer_median.transform(X_tr[:, other_mask])
        out_te[:, other_mask] = imputer_median.transform(X_te[:, other_mask])
        out_oo[:, other_mask] = imputer_median.transform(X_oo[:, other_mask])
    if consec_mask.any():
        imputer_zero.fit(X_tr[:, consec_mask])
        out_tr[:, consec_mask] = imputer_zero.transform(X_tr[:, consec_mask])
        out_te[:, consec_mask] = imputer_zero.transform(X_te[:, consec_mask])
        out_oo[:, consec_mask] = imputer_zero.transform(X_oo[:, consec_mask])
    return out_tr, out_te, out_oo

print("\n[08B] Imputing (train-fit only)...")
X_train_imp, X_test_imp, X_oos_imp = impute_splits(
    X_train_raw, X_test_raw, X_oos_raw
)
assert not np.isnan(X_train_imp).any(), "NAs remain after imputation — train"
assert not np.isnan(X_test_imp).any(),  "NAs remain after imputation — test"
assert not np.isnan(X_oos_imp).any(),   "NAs remain after imputation — OOS"
print("  Imputation complete — 0 NAs remaining")

# PIT normalisation
print("\n[08B] Applying PIT normalisation (normal distribution, train-fit only)...")
pit = QuantileTransformer(
    output_distribution="normal",
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
print(f"  Train mean (should be ≈ 0): {X_train_norm[:, :n_cont].mean():.4f}")
print(f"  Train std  (should be ≈ 1): {X_train_norm[:, :n_cont].std():.4f}")

# Validation split
y_strat = np.where(np.isnan(y_train_full), 0.0, y_train_full)

train_idx, val_idx = train_test_split(
    np.arange(len(X_train_norm)),
    test_size=CFG["val_fraction"],
    random_state=CFG["seed"],
    stratify=y_strat
)

X_vae_train = X_train_norm[train_idx]
X_vae_val   = X_train_norm[val_idx]
y_vae_train = y_train_full[train_idx]
y_vae_val   = y_train_full[val_idx]

print(f"\n[08B] VAE train/val split:")
print(f"  VAE train : {len(X_vae_train):,} rows")
print(f"  VAE val   : {len(X_vae_val):,} rows")

# PyTorch tensors
X_vae_train_t = torch.tensor(X_vae_train, dtype=torch.float32)
X_vae_val_t   = torch.tensor(X_vae_val,   dtype=torch.float32)
X_train_t     = torch.tensor(X_train_norm, dtype=torch.float32)
X_test_t      = torch.tensor(X_test_norm,  dtype=torch.float32)
X_oos_t       = torch.tensor(X_oos_norm,   dtype=torch.float32)
y_vae_train_t = torch.tensor(y_vae_train,  dtype=torch.float32)
y_vae_val_t   = torch.tensor(y_vae_val,    dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_vae_train_t, y_vae_train_t),
    batch_size=CFG["batch_size"],
    shuffle=True,
    drop_last=False
)
val_loader = DataLoader(
    TensorDataset(X_vae_val_t, y_vae_val_t),
    batch_size=CFG["batch_size"],
    shuffle=False
)

print(f"\n[08B] DataLoaders:")
print(f"  Train batches : {len(train_loader)}")
print(f"  Val batches   : {len(val_loader)}")

# ==============================================================================
# 8. Model Architecture
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, z_dim: int):
        super().__init__()
        layers = []
        in_d   = input_dim
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
    def __init__(self, z_dim: int, hidden_dims: list,
                 n_cont: int, n_binary: int):
        super().__init__()
        layers = []
        in_d   = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        self.net      = nn.Sequential(*layers)
        self.out_cont = nn.Linear(in_d, n_cont)   if n_cont   > 0 else None
        self.out_bin  = nn.Linear(in_d, n_binary) if n_binary > 0 else None

    def forward(self, z):
        h     = self.net(z)
        parts = []
        if self.out_cont is not None:
            parts.append(self.out_cont(h))
        if self.out_bin is not None:
            parts.append(torch.sigmoid(self.out_bin(h)))
        return torch.cat(parts, dim=1)


class ClassifierHead(nn.Module):
    def __init__(self, z_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        in_d   = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.GELU()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_mean):
        return self.net(z_mean).squeeze(-1)


class BetaVAE(nn.Module):
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
        std = torch.exp(0.5 * z_lv)
        eps = torch.randn_like(std)
        return z_mu + eps * std

    def forward(self, x):
        z_mu, z_lv = self.encoder(x)
        z          = self.reparameterise(z_mu, z_lv)
        x_recon    = self.decoder(z)
        y_logit    = self.classifier(z_mu)
        return x_recon, z_mu, z_lv, y_logit

    def compute_loss(self, x, x_recon, z_mu, z_lv,
                     y_true, beta, gamma, labelled_mask):
        if self.n_cont > 0:
            mse = F.mse_loss(
                x_recon[:, :self.n_cont],
                x[:, :self.n_cont],
                reduction="none"
            ).sum(dim=1).mean()
        else:
            mse = torch.tensor(0.0, device=x.device)

        if self.n_binary > 0:
            bce_recon = F.binary_cross_entropy(
                x_recon[:, self.n_cont:],
                x[:, self.n_cont:],
                reduction="none"
            ).sum(dim=1).mean()
        else:
            bce_recon = torch.tensor(0.0, device=x.device)

        recon_loss = mse + bce_recon

        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_lv - z_mu.pow(2) - z_lv.exp(), dim=1)
        )

        if gamma > 0 and labelled_mask.sum() > 0:
            y_logit    = self.classifier(z_mu)
            logits_lab = y_logit[labelled_mask]
            y_true_lab = y_true[labelled_mask]
            clf_loss   = F.binary_cross_entropy_with_logits(
                logits_lab, y_true_lab
            )
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

    @torch.no_grad()
    def eval_loss(self, x, y, beta, gamma):
        self.eval()
        x_recon, z_mu, z_lv, y_logit = self(x)
        lab_mask = ~torch.isnan(y)
        losses   = self.compute_loss(
            x, x_recon, z_mu, z_lv, y,
            beta=beta, gamma=gamma, labelled_mask=lab_mask
        )
        self.train()
        return losses


# ==============================================================================
# 9. Training
# ==============================================================================

def get_beta(epoch: int, max_beta: float, warmup: int) -> float:
    if warmup <= 0:
        return max_beta
    return min(max_beta, max_beta * (epoch + 1) / warmup)


def train_vae(model, train_loader, val_loader, cfg, device):
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    history = {k: [] for k in
               ["train_total", "train_recon", "train_kl", "train_clf",
                "val_total",   "val_recon",   "val_kl",   "val_clf"]}

    best_val_loss = float("inf")
    best_state    = None
    patience_ct   = 0

    model.train()

    for epoch in range(cfg["epochs"]):

        beta_now     = get_beta(epoch, cfg["beta"], cfg["kl_warmup"])
        epoch_losses = {k: 0.0 for k in ["total", "recon", "kl", "clf"]}
        n_batches    = 0

        model.train()
        for x_batch, y_batch in train_loader:
            x_batch  = x_batch.to(device)
            y_batch  = y_batch.to(device)
            lab_mask = ~torch.isnan(y_batch)

            optimiser.zero_grad()
            x_recon, z_mu, z_lv, _ = model(x_batch)
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

        for k in epoch_losses:
            history[f"train_{k}"].append(epoch_losses[k] / n_batches)

        val_losses    = {k: 0.0 for k in ["total", "recon", "kl", "clf"]}
        n_val_batches = 0

        with torch.no_grad():
            model.eval()
            for x_val, y_val in val_loader:
                x_val    = x_val.to(device)
                y_val    = y_val.to(device)
                lab_mask = ~torch.isnan(y_val)

                x_recon, z_mu, z_lv, _ = model(x_val)
                v_losses = model.compute_loss(
                    x_val, x_recon, z_mu, z_lv, y_val,
                    beta=beta_now, gamma=cfg["gamma"],
                    labelled_mask=lab_mask
                )
                for k in val_losses:
                    val_losses[k] += v_losses[k].item()
                n_val_batches += 1

        for k in val_losses:
            history[f"val_{k}"].append(val_losses[k] / n_val_batches)

        scheduler.step(history["val_total"][-1])

        if epoch % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{cfg['epochs']} | "
                f"β={beta_now:.3f} | "
                f"train={history['train_total'][-1]:.2f} | "
                f"val={history['val_total'][-1]:.2f} | "
                f"kl={history['train_kl'][-1]:.2f} | "
                f"clf={history['train_clf'][-1]:.4f}"
            )

        if history["val_total"][-1] < best_val_loss - 1e-3:
            best_val_loss = history["val_total"][-1]
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_ct   = 0
        else:
            patience_ct += 1
            if patience_ct >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best val: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Best weights restored (val loss: {best_val_loss:.4f})")

    return history


# ==============================================================================
# 10. Encoding
# ==============================================================================

@torch.no_grad()
def encode(model, X_tensor, device, batch_size=1024):
    model.eval()
    z_list   = []
    err_list = []

    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
    for (x_batch,) in loader:
        x_batch    = x_batch.to(device)
        z_mu, z_lv = model.encoder(x_batch)
        z_samp     = model.reparameterise(z_mu, z_lv)
        x_recon    = model.decoder(z_mu)
        err        = F.mse_loss(x_recon, x_batch, reduction="none").mean(dim=1)
        z_list.append(z_mu.cpu().numpy())
        err_list.append(err.cpu().numpy())

    return np.vstack(z_list), np.concatenate(err_list)


# ==============================================================================
# 11. Diagnostics & Plots
# ==============================================================================

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(history["train_total"], label="Train total", linewidth=1.5)
    axes[0].plot(history["val_total"],   label="Val total",   linewidth=1.5,
                 linestyle="--")
    axes[0].set_title("Total Loss (Train vs Val)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_recon"], label="Train recon", linewidth=1.5)
    axes[1].plot(history["val_recon"],   label="Val recon",   linewidth=1.5,
                 linestyle="--")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["train_kl"],  label="KL",  linewidth=1.5,
                 color="orange")
    axes[2].plot(history["train_clf"], label="Clf", linewidth=1.5,
                 color="red")
    axes[2].set_title("KL & Classification Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_latent_space(z_means, y_labels, save_path, n_pairs=4):
    valid  = ~np.isnan(y_labels)
    z_v    = z_means[valid]
    y_v    = y_labels[valid]
    n_dims = z_means.shape[1]

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
    n_active    = int((dim_var >= 1e-3).sum())
    n_collapsed = int((dim_var < 1e-3).sum())

    print(f"\n  Latent space diagnostics:")
    print(f"    z_dim              : {z_train.shape[1]}")
    print(f"    Active dims        : {n_active}  (var >= 1e-3)")
    print(f"    Collapsed dims     : {n_collapsed}  (var < 1e-3)")
    print(f"    Dim variance range : [{dim_var.min():.4f}, "
          f"{dim_var.max():.4f}]")
    print(f"    Target (good run)  : <= 3 collapsed dims")

    valid      = ~np.isnan(y_train)
    csi_err    = err_train[valid & (y_train == 1)].mean()
    noncsi_err = err_train[valid & (y_train == 0)].mean()
    ratio      = csi_err / noncsi_err if noncsi_err > 0 else float("nan")

    print(f"    Recon err CSI      : {csi_err:.4f}")
    print(f"    Recon err Non-CSI  : {noncsi_err:.4f}")
    print(f"    Ratio (CSI/Non-CSI): {ratio:.3f}  "
          f"(>1.0 = VAE learned anomaly signal)")
    print(f"    Test recon p95     : {np.percentile(err_test, 95):.4f}")

    if n_collapsed > 5:
        print(f"    ⚠ >5 collapsed dims — consider reducing beta "
              f"(current: {CFG['beta']})")
    if ratio < 1.0:
        print(f"    ⚠ CSI recon ratio < 1.0 — VAE not learning anomaly "
              f"signal. Check gamma and training convergence.")


# ==============================================================================
# 12. Main
# ==============================================================================

def main():

    print(f"\n{'='*60}")
    print(f"[08B] Beta-VAE v3 — {vcfg['description']}")
    print(f"{'='*60}")
    print(f"  Input features : {n_features}")
    print(f"  Continuous     : {n_cont}")
    print(f"  Binary         : {n_binary}  {bin_cols}")
    print(f"  z_dim          : {CFG['z_dim']}")
    print(f"  beta           : {CFG['beta']}")
    print(f"  gamma          : {CFG['gamma']}  "
          f"({'supervised' if CFG['gamma'] > 0 else 'pure VAE'})")
    print(f"  val_fraction   : {CFG['val_fraction']}")
    print(f"  Output         : {PATH_FEATURES_LATENT.name}")

    # Build model
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

    # Train
    print(f"\n[08B] Training [{VAE_INPUT.upper()}]...")
    history = train_vae(model, train_loader, val_loader, CFG, DEVICE)

    # Save model weights — separate dirs per VAE_INPUT
    torch.save(model.state_dict(),
               DIR_MODELS_RUN / "vae_weights.pt")
    torch.save(model.encoder.state_dict(),
               DIR_MODELS_RUN / "encoder_weights.pt")

    cfg_save = {
        **CFG,
        "vae_input"  : VAE_INPUT,
        "description": vcfg["description"],
        "n_features" : n_features,
        "n_cont"     : n_cont,
        "n_binary"   : n_binary,
        "col_order"  : col_order,
        "binary_cols": bin_cols,
        "pit_dist"   : "normal",
        "output_file": str(PATH_FEATURES_LATENT),
    }
    with open(DIR_MODELS_RUN / "vae_config.json", "w") as f:
        json.dump(cfg_save, f, indent=2)

    print(f"\n[08B] Model saved to: {DIR_MODELS_RUN}")

    # Encode all splits
    print("\n[08B] Encoding train / test / OOS...")
    z_train, err_train = encode(model, X_train_t, DEVICE)
    z_test,  err_test  = encode(model, X_test_t,  DEVICE)
    z_oos,   err_oos   = encode(model, X_oos_t,   DEVICE)

    # Diagnostics
    print_latent_diagnostics(z_train, y_train_full,
                             z_test, err_train, err_test)

    fig_suffix = vcfg["fig_suffix"]
    plot_training_curves(
        history,
        DIR_FIGURES / f"vae_training_curves_{fig_suffix}.png"
    )
    plot_latent_space(
        z_train, y_train_full,
        DIR_FIGURES / f"vae_latent_space_{fig_suffix}.png"
    )

    # Assemble features_latent
    z_cols = [f"z{i+1}" for i in range(CFG["z_dim"])]

    def build_latent_df(src_df, z_arr, err_arr, split_name):
        id_part = src_df[["permno", "year", "y", "censored"]].reset_index(
            drop=True)
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
    print(f"  Output : {PATH_FEATURES_LATENT}")

    # Save
    features_latent.to_parquet(PATH_FEATURES_LATENT, index=False)
    print(f"\n[08B] Saved: {PATH_FEATURES_LATENT}")

    # Assertions
    latent_cols = z_cols + ["vae_recon_error"]

    assert features_latent[latent_cols].isna().sum().sum() == 0, \
        "NAs in latent features — check encoding"
    assert features_latent[["permno", "year"]].isna().sum().sum() == 0, \
        "NAs in identifiers"
    assert features_latent["split"].nunique() == 3, \
        "Expected train/test/oos in split column"

    n_collapsed = int(
        (np.array([features_latent[c].var() for c in z_cols]) < 1e-3).sum()
    )
    print(f"\n[08B] Final collapsed dims: {n_collapsed} / {CFG['z_dim']}")
    if n_collapsed > 5:
        print(f"  Consider reducing beta from {CFG['beta']} to "
              f"{CFG['beta'] * 0.5:.2f} and retraining.")

    print("\n[08B] All assertions passed.")
    print(f"[08B] DONE [{VAE_INPUT.upper()}] → {PATH_FEATURES_LATENT.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()