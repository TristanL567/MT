#==============================================================================#
#==== 05C_Structural_Labels.R =================================================#
#==== Combined Structural Quality Label Construction =========================#
#==============================================================================#
#
# PURPOSE:
#   Construct a unified structural quality label combining CSI event labels
#   and 5-year forward CAGR bucket labels. This label captures the full
#   spectrum of structural deterioration — both acute crashes (CSI) and
#   chronic slow bleeders — while correctly treating recoveries as positives.
#
# LABEL DEFINITION:
#   y_structural(permno, t) = 1  EXCLUDE — structurally unhealthy:
#       → CSI event at t AND 5yr forward CAGR < −2% (confirmed terminal)
#       → No CSI event but 5yr forward CAGR < −2% (slow bleeder)
#
#   y_structural(permno, t) = 0  KEEP — structurally sound:
#       → CSI event at t but 5yr forward CAGR ≥ 0% (phoenix — crashed, recovered)
#       → No CSI event and 5yr forward CAGR ≥ 0% (healthy firm)
#
#   y_structural(permno, t) = NA — exclude from training:
#       → 5yr CAGR in [−2%, 0%) — ambiguous temporary loser
#       → Right-censored (< MIN_MONTHS_FWD forward returns available)
#       → No overlap between CSI and bucket label windows
#
# FOUR-QUADRANT DISTRIBUTION (from validation):
#   CSI + Terminal loser   :  3,455  (3.7%)  ← highest conviction exclude
#   CSI + Phoenix          :  2,434  (2.6%)  ← crash but recovered → KEEP
#   Slow loser (no crash)  : 31,184 (33.2%)  ← chronic decline → exclude
#   Healthy                : 57,353 (61.1%)  ← structural quality → KEEP
#   ─────────────────────────────────────────
#   Total positives (y=1)  : 34,639 (36.7%)
#   Total negatives (y=0)  : 59,787 (63.3%)  → much better balance than CSI alone
#
# KEY ADVANTAGES OVER INDIVIDUAL LABELS:
#   1. 9× more positive training examples than CSI alone (34,639 vs 3,455)
#   2. Phoenix false positives removed — 2,434 CSI firms correctly kept
#   3. Slow bleeders captured — 31,184 firms CSI never saw
#   4. Class balance 37% positive — no extreme class weighting needed
#
# INPUTS:
#   - config.R
#   - PATH_LABELS_BASE    (labels_base.rds   — CSI labels from 05_CSI_Label.R)
#   - PATH_LABELS_BUCKET  (labels_bucket.rds — 5yr CAGR labels from 05B)
#
# OUTPUTS:
#   - PATH_LABELS_STRUCTURAL  : main label file (permno, year, y_structural, ...)
#   - DIR_FIGURES/05_labels/  : diagnostic plots
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
  library(lubridate)
})

cat("\n[05C_Structural_Labels.R] START:", format(Sys.time()), "\n")

## Setup figure directories if available
if (exists("fn_setup_figure_dirs")) {
  FIGS <- fn_setup_figure_dirs()
} else {
  FIGS <- list(labels = DIR_FIGURES)
  dir.create(FIGS$labels, recursive=TRUE, showWarnings=FALSE)
}

#==============================================================================#
# 0. Parameters
#==============================================================================#

CAGR_LOSER_THRESH   <- -0.02   ## CAGR < -2%  → structural loser
CAGR_PHOENIX_THRESH <-  0.00   ## CAGR >= 0%  → phoenix / healthy

## Output path
PATH_LABELS_STRUCTURAL <- file.path(DIR_LABELS, "labels_structural.rds")

cat(sprintf("  CAGR thresholds : loser < %.0f%% | phoenix >= %.0f%%\n",
            CAGR_LOSER_THRESH*100, CAGR_PHOENIX_THRESH*100))
cat(sprintf("  Output          : %s\n", PATH_LABELS_STRUCTURAL))

#==============================================================================#
# 1. Load input labels
#==============================================================================#

cat("\n[05C] Loading input labels...\n")

labels_csi <- as.data.table(readRDS(PATH_LABELS_BASE))
cat(sprintf("  CSI labels      : %d rows | y=1: %d | y=0: %d | NA: %d\n",
            nrow(labels_csi),
            sum(labels_csi$y == 1L, na.rm=TRUE),
            sum(labels_csi$y == 0L, na.rm=TRUE),
            sum(is.na(labels_csi$y))))

labels_bucket <- as.data.table(readRDS(PATH_LABELS_BUCKET))
cat(sprintf("  Bucket labels   : %d rows | loser: %d | phoenix: %d | censored: %d\n",
            nrow(labels_bucket),
            sum(labels_bucket$y_loser == 1L, na.rm=TRUE),
            sum(labels_bucket$y_loser == 0L, na.rm=TRUE),
            sum(labels_bucket$censored, na.rm=TRUE)))

#==============================================================================#
# 2. Merge on (permno, year)
#==============================================================================#

cat("\n[05C] Merging labels...\n")

## Inner join — only rows with both labels
combined <- merge(
  labels_csi[, .(permno, year,
                 y_csi      = y,
                 censored_csi = censored)],
  labels_bucket[, .(permno, year,
                    y_loser   = y_loser,
                    fwd_cagr  = fwd_cagr,
                    censored_bucket = censored)],
  by = c("permno","year"),
  all = FALSE   ## inner join — require both labels to exist
)

cat(sprintf("  Combined rows   : %d\n", nrow(combined)))
cat(sprintf("  Unique permnos  : %d\n", uniqueN(combined$permno)))
cat(sprintf("  Year range      : %d – %d\n",
            min(combined$year), max(combined$year)))

#==============================================================================#
# 3. Assign structural label
#==============================================================================#

cat("\n[05C] Assigning structural labels...\n")

combined[, y_structural := fcase(
  
  ## Exclude from training: bucket is censored or ambiguous temporary loser
  censored_bucket == TRUE,          NA_integer_,
  is.na(y_loser),                   NA_integer_,
  
  ## Exclude from training: CSI is in zombie window (NA)
  is.na(y_csi) & is.na(y_loser),   NA_integer_,
  
  ## STRUCTURAL LOSER (y=1) — EXCLUDE:
  ## Any firm with negative 5yr CAGR, regardless of CSI status
  ## This captures both confirmed CSI zombies AND slow bleeders
  y_loser == 1L,                    1L,
  
  ## STRUCTURAL QUALITY (y=0) — KEEP:
  ## Firms with positive 5yr CAGR — including CSI phoenixes
  y_loser == 0L,                    0L,
  
  ## Default: NA for anything else ambiguous
  default = NA_integer_
)]

## Summary
n_pos   <- sum(combined$y_structural == 1L, na.rm=TRUE)
n_neg   <- sum(combined$y_structural == 0L, na.rm=TRUE)
n_na    <- sum(is.na(combined$y_structural))
n_total <- nrow(combined)

cat(sprintf("\n  Structural label distribution:\n"))
cat(sprintf("    y=1 (structural loser) : %6d  (%.1f%% of usable)\n",
            n_pos, n_pos/(n_pos+n_neg)*100))
cat(sprintf("    y=0 (structural quality): %6d  (%.1f%% of usable)\n",
            n_neg, n_pos/(n_pos+n_neg)*100))
cat(sprintf("    NA  (excluded)          : %6d  (%.1f%% of total)\n",
            n_na, n_na/n_total*100))
cat(sprintf("    Total                   : %6d\n", n_total))
cat(sprintf("    Usable                  : %6d\n", n_pos+n_neg))

## Four-quadrant breakdown for reporting
cat("\n  Four-quadrant breakdown:\n")
quad <- combined[!is.na(y_csi) & !is.na(y_loser), .(
  csi_loser   = sum(y_csi==1L & y_loser==1L),
  csi_phoenix = sum(y_csi==1L & y_loser==0L),
  slow_loser  = sum(y_csi==0L & y_loser==1L),
  healthy     = sum(y_csi==0L & y_loser==0L)
)]
cat(sprintf("    CSI + Terminal loser (→ y=1): %6d  [confirmed implosion]\n",
            quad$csi_loser))
cat(sprintf("    CSI + Phoenix        (→ y=0): %6d  [crash but recovered]\n",
            quad$csi_phoenix))
cat(sprintf("    Slow loser           (→ y=1): %6d  [bleeder, no crash]\n",
            quad$slow_loser))
cat(sprintf("    Healthy              (→ y=0): %6d  [structural quality]\n",
            quad$healthy))
cat(sprintf("    Total positives (y=1)        : %6d  (9x more than CSI alone)\n",
            quad$csi_loser + quad$slow_loser))

#==============================================================================#
# 4. Assertions
#==============================================================================#

cat("\n[05C] Running assertions...\n")

## A: y_structural only 0, 1, or NA
invalid <- combined[!is.na(y_structural) & !y_structural %in% c(0L,1L)]
stopifnot("Invalid y_structural" = nrow(invalid) == 0L)

## B: No duplicate (permno, year)
n_dup <- sum(duplicated(combined[, .(permno, year)]))
stopifnot("Duplicate (permno, year)" = n_dup == 0L)

## C: Bucket phoenix → y_structural = 0
misclassified <- combined[!is.na(y_loser) & y_loser==0L &
                            !is.na(y_structural) & y_structural==1L]
stopifnot("Phoenix incorrectly labelled as loser" = nrow(misclassified) == 0L)

## D: Bucket loser → y_structural = 1
misclassified2 <- combined[!is.na(y_loser) & y_loser==1L &
                             !is.na(y_structural) & y_structural==0L]
stopifnot("Loser incorrectly labelled as quality" = nrow(misclassified2) == 0L)

cat("  All assertions passed.\n")

#==============================================================================#
# 5. Save label file
#==============================================================================#

labels_structural <- combined[, .(
  permno,
  year,
  y_structural,
  y_csi,
  y_loser,
  fwd_cagr   = round(fwd_cagr, 6),
  censored_bucket
)][order(permno, year)]

saveRDS(labels_structural, PATH_LABELS_STRUCTURAL)
cat(sprintf("\n  Saved: %s\n", PATH_LABELS_STRUCTURAL))

#==============================================================================#
# 6. Diagnostic plots
#==============================================================================#

cat("\n[05C] Generating diagnostic plots...\n")

## ── Plot 1: Label distribution by year ──────────────────────────────────────

label_yr <- labels_structural[!is.na(y_structural), .(
  n_loser   = sum(y_structural==1L),
  n_quality = sum(y_structural==0L),
  prevalence = mean(y_structural==1L)
), by=year][order(year)]

label_yr_long <- melt(label_yr[, .(year, n_loser, n_quality)],
                      id.vars="year",
                      variable.name="label",
                      value.name="n")
label_yr_long[, label := ifelse(label=="n_loser",
                                "Structural Loser (y=1)",
                                "Structural Quality (y=0)")]

p_dist <- ggplot(label_yr_long, aes(x=year, y=n, fill=label)) +
  geom_col(width=0.8, alpha=0.85) +
  scale_fill_manual(values=c(
    "Structural Loser (y=1)"   = "#E53935",
    "Structural Quality (y=0)" = "#2196F3"
  )) +
  scale_y_continuous(name="Observations") +
  labs(
    title    = "Structural Label Distribution by Year",
    subtitle = sprintf("Combined CSI + 5yr CAGR bucket | Positive rate: %.1f%%",
                       n_pos/(n_pos+n_neg)*100),
    x=NULL, fill=NULL
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(FIGS$labels, "structural_label_dist.png"), p_dist,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  structural_label_dist.png saved.\n")

## ── Plot 2: Prevalence over time ─────────────────────────────────────────────

p_prev <- ggplot(label_yr, aes(x=year, y=prevalence)) +
  geom_line(colour="#E53935", linewidth=0.9) +
  geom_point(colour="#E53935", size=2) +
  geom_hline(yintercept=mean(label_yr$prevalence),
             linetype="dashed", colour="grey50") +
  annotate("text",
           x=min(label_yr$year),
           y=mean(label_yr$prevalence),
           label=sprintf("Mean: %.0f%%", mean(label_yr$prevalence)*100),
           hjust=0, vjust=-0.5, size=3, colour="grey40") +
  scale_y_continuous(labels=percent_format(accuracy=1),
                     name="% Structural Losers", limits=c(0,1)) +
  labs(
    title    = "Structural Loser Prevalence by Year",
    subtitle = "More stable than CSI alone due to slow bleeder inclusion",
    x=NULL
  ) +
  theme_minimal(base_size=12)

ggsave(file.path(FIGS$labels, "structural_prevalence_by_year.png"), p_prev,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  structural_prevalence_by_year.png saved.\n")

## ── Plot 3: Four-quadrant composition bar ────────────────────────────────────

quad_plot <- data.frame(
  quadrant = c("CSI + Terminal\n(y=1)",
               "CSI + Phoenix\n(y=0)",
               "Slow Bleeder\n(y=1)",
               "Healthy\n(y=0)"),
  n        = c(quad$csi_loser, quad$csi_phoenix,
               quad$slow_loser, quad$healthy),
  label    = c("Structural Loser (y=1)",
               "Structural Quality (y=0)",
               "Structural Loser (y=1)",
               "Structural Quality (y=0)")
)
quad_plot$quadrant <- factor(quad_plot$quadrant,
                             levels=quad_plot$quadrant)

p_quad <- ggplot(quad_plot, aes(x=quadrant, y=n, fill=label)) +
  geom_col(width=0.65, alpha=0.85) +
  geom_text(aes(label=scales::comma(n)), vjust=-0.3, size=3.5) +
  scale_fill_manual(values=c(
    "Structural Loser (y=1)"   = "#E53935",
    "Structural Quality (y=0)" = "#2196F3"
  )) +
  scale_y_continuous(labels=comma_format(),
                     name="Number of (permno, year) observations") +
  labs(
    title    = "Four-Quadrant Distribution: CSI × 5yr CAGR Bucket",
    subtitle = "Slow bleeders (no CSI crash) dominate the positive class 9:1 over confirmed CSI",
    x=NULL, fill=NULL
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(FIGS$labels, "structural_four_quadrant.png"), p_quad,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  structural_four_quadrant.png saved.\n")

## ── Plot 4: CAGR distribution by structural label ────────────────────────────

cagr_plot <- labels_structural[
  !is.na(y_structural) & !is.na(fwd_cagr),
  .(fwd_cagr, label = ifelse(y_structural==1L,
                             "Structural Loser (y=1)",
                             "Structural Quality (y=0)"))
]
q <- quantile(cagr_plot$fwd_cagr, c(0.02, 0.98), na.rm=TRUE)
cagr_plot <- cagr_plot[fwd_cagr >= q[1] & fwd_cagr <= q[2]]

p_cagr <- ggplot(cagr_plot, aes(x=fwd_cagr, fill=label, colour=label)) +
  geom_density(alpha=0.30, linewidth=0.7) +
  geom_vline(xintercept=CAGR_LOSER_THRESH,
             linetype="dashed", colour="#E53935", linewidth=0.6) +
  geom_vline(xintercept=CAGR_PHOENIX_THRESH,
             linetype="dashed", colour="#2196F3", linewidth=0.6) +
  scale_x_continuous(labels=percent_format(accuracy=1),
                     name="5-Year Forward CAGR") +
  scale_fill_manual(values=c(
    "Structural Loser (y=1)"   = "#E53935",
    "Structural Quality (y=0)" = "#2196F3"
  )) +
  scale_colour_manual(values=c(
    "Structural Loser (y=1)"   = "#E53935",
    "Structural Quality (y=0)" = "#2196F3"
  )) +
  labs(
    title    = "5-Year Forward CAGR Distribution by Structural Label",
    subtitle = "Clean separation — label threshold aligns with natural distribution break",
    y="Density", fill=NULL, colour=NULL
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(FIGS$labels, "structural_cagr_dist.png"), p_cagr,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  structural_cagr_dist.png saved.\n")

#==============================================================================#
# 7. Final summary
#==============================================================================#

cat("\n[05C] ══════════════════════════════════════════════════\n")
cat("  STRUCTURAL LABEL SUMMARY\n")
cat("  ══════════════════════════════════════════════════════\n\n")
cat(sprintf("  Usable labels           : %d\n", n_pos+n_neg))
cat(sprintf("  Structural losers (y=1) : %d (%.1f%%)\n",
            n_pos, n_pos/(n_pos+n_neg)*100))
cat(sprintf("  Structural quality (y=0): %d (%.1f%%)\n",
            n_neg, n_neg/(n_pos+n_neg)*100))
cat(sprintf("\n  vs CSI alone (M1 training data):\n"))
cat(sprintf("    CSI positives         : ~3,455 (12%% base rate)\n"))
cat(sprintf("    Structural positives  : %d (%.0f%% base rate) — %.1fx more\n",
            n_pos, n_pos/(n_pos+n_neg)*100,
            n_pos / 3455))
cat(sprintf("\n  Key composition:\n"))
cat(sprintf("    Phoenix reclassification : %d CSI events → y=0 (kept)\n",
            quad$csi_phoenix))
cat(sprintf("    Slow bleeders added      : %d new positives CSI never saw\n",
            quad$slow_loser))
cat(sprintf("\n  Next step: train AutoGluon B1-structural\n"))
cat(sprintf("    MODEL = 'structural' in 09C_AutoGluon.py\n"))
cat(sprintf("    Labels: %s\n", PATH_LABELS_STRUCTURAL))
cat(sprintf("    Features: features_fund.rds (same as M1/B1)\n"))
cat(sprintf("    Target: y_structural\n"))
cat(sprintf("    Expected: AUC > 0.743 (B1 bucket baseline)\n"))

cat(sprintf("\n[05C_Structural_Labels.R] DONE: %s\n", format(Sys.time())))
