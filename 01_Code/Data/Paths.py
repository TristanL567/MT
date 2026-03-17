from pathlib import Path

DATA_ROOT    = Path(r"C:\Users\Tristan Leiter\Documents\MT")
DIR_DATA     = DATA_ROOT / "02_Data"
DIR_FEATURES = DIR_DATA  / "Features"

PATH_FEATURES_RAW = DIR_FEATURES / "features_raw.rds"
PATH_SPLITS       = DIR_FEATURES / "splits.rds"

print(PATH_FEATURES_RAW.exists())
print(PATH_SPLITS.exists())
print(list(DIR_FEATURES.glob("*.rds")))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path(r"C:\Users\Tristan Leiter\Documents\MT")
DIR_DATA     = DATA_ROOT / "02_Data"
DIR_FEATURES = DIR_DATA  / "Features"
DIR_FIGURES  = DATA_ROOT / "03_Output" / "Figures"
DIR_MODELS   = DATA_ROOT / "03_Output" / "Models" / "VAE"

DIR_FIGURES.mkdir(parents=True, exist_ok=True)
DIR_MODELS.mkdir(parents=True,  exist_ok=True)

PATH_FEATURES_RAW    = DIR_FEATURES / "features_raw.rds"
PATH_SPLITS          = DIR_FEATURES / "splits.rds"
PATH_FEATURES_LATENT = DIR_FEATURES / "features_latent.parquet"

print(PATH_FEATURES_RAW.exists())   # Should print True
print(PATH_SPLITS.exists())          # Should print True
print(list(DIR_FEATURES.glob("*.rds")))  # Should list your files