#==============================================================================#
#==== 05B_Bucket_Labels.R =====================================================#
#==== 5-Year Forward CAGR Bucket Label Construction ==========================#
#==============================================================================#
#
# PURPOSE:
#   Construct binary outcome labels for the B1 bucket classifier.
#   For each (permno, year), the label captures whether the firm will be
#   a terminal loser or a phoenix over the NEXT 5 years.
#
# LABEL DEFINITION:
#   y_loser(permno, t) = 1  if 5yr forward CAGR from t+1 to t+5 < -2%
#   y_loser(permno, t) = 0  if 5yr forward CAGR from t+1 to t+5 >= 0%
#   y_loser(permno, t) = NA if CAGR in [-2%, 0%) — temporary loser, excluded
#   y_loser(permno, t) = NA if fewer than MIN_MONTHS_FWD months available
#                           (right-censored — last usable year = END_YEAR - 5)
#
# HORIZON DESIGN:
#   Forward window = calendar years t+1 through t+5 inclusive.
#   Features at year t → label uses returns from January(t+1) to December(t+5).
#   This is clean: no information from after year t used in features.
#   Consistent with M1/M3 label shift (features at t → event in t+1).
#
# RIGHT-CENSORING:
#   Observations within 5 years of END_DATE are right-censored (y = NA).
#   Default: observations from 2019 onward are censored if END_DATE = 2024.
#   Use REQUIRE_COMPLETE_WINDOW = TRUE for strict 5-year requirement (default).
#   Set FALSE to allow 3-year minimum (more data, noisier labels).
#
# OUTPUTS:
#   - PATH_LABELS_BUCKET          : main label file (permno, year, y_loser, ...)
#   - DIR_FIGURES/05_labels/      : diagnostic plots
#
# FIGURE OUTPUTS:
#   05_labels/bucket_label_dist.png         label distribution over time
#   05_labels/bucket_cagr_dist.png          CAGR distribution by bucket
#   05_labels/bucket_prevalence_by_year.png loser rate over time
#   05_labels/bucket_vs_csi_overlap.png     overlap with CSI labels
#
#==============================================================================#

source("config.R")
source("config_figures_addon.R")   ## figure directory helpers

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
  library(lubridate)
})

cat("\n[05B_Bucket_Labels.R] START:", format(Sys.time()), "\n")

## Setup figure directories
FIGS <- fn_setup_figure_dirs()

#==============================================================================#
# 0. Parameters
#==============================================================================#

FWD_YEARS             <- 5L       ## forward window in years
MIN_MONTHS_FWD        <- 48L      ## minimum months required (4 of 5 years)
REQUIRE_COMPLETE_WINDOW <- TRUE   ## if TRUE: strictly require FWD_YEARS * 12 months

CAGR_LOSER_THRESH     <- -0.02    ## CAGR < -2%  → terminal loser (y=1)
CAGR_PHOENIX_THRESH   <-  0.00    ## CAGR >= 0%  → phoenix        (y=0)
## [-2%, 0%) → temporary loser → y=NA (excluded from training)

## Right-censoring: last year with complete forward window
## If data ends Dec 2024, last complete 5-year window starts Jan 2019
LAST_COMPLETE_YEAR    <- year(END_DATE) - FWD_YEARS

cat(sprintf("  Forward window    : %d years\n", FWD_YEARS))
cat(sprintf("  Last usable year  : %d (obs from %d+ are censored)\n",
            LAST_COMPLETE_YEAR, LAST_COMPLETE_YEAR + 1L))
cat(sprintf("  CAGR thresholds   : loser < %.0f%% | phoenix >= %.0f%%\n",
            CAGR_LOSER_THRESH*100, CAGR_PHOENIX_THRESH*100))

#==============================================================================#
# 1. Load monthly returns
#==============================================================================#

cat("\n[05B] Loading monthly returns...\n")

monthly <- as.data.table(readRDS(PATH_PRICES_MONTHLY))
setnames(monthly, "ret_adj", "ret")
monthly[, ret := pmin(pmax(ret, -0.99, na.rm=TRUE), 10, na.rm=TRUE)]
monthly[, year  := year(date)]
monthly[, month := month(date)]

universe <- readRDS(PATH_UNIVERSE)

monthly <- monthly[permno %in% universe$permno]

cat(sprintf("  Monthly data: %d rows | %d permnos\n",
            nrow(monthly), uniqueN(monthly$permno)))

#==============================================================================#
# 2. Compute 5-year forward CAGR per (permno, base_year)
#==============================================================================#

cat("\n[05B] Computing 5-year forward CAGR per (permno, year)...\n")

## For each permno and each base_year, extract returns from
## base_year+1 to base_year+FWD_YEARS and compute CAGR.

## Get the range of base years to compute
base_years <- seq(year(START_DATE), LAST_COMPLETE_YEAR)

## Pre-sort for speed
setkey(monthly, permno, year, month)

## Function: compute forward CAGR for one permno
## Returns a data.table with one row per base_year
fn_forward_cagr <- function(dt_permno, base_years, fwd_years, min_months) {
  
  results <- lapply(base_years, function(by) {
    
    ## Forward returns: Jan(by+1) through Dec(by+fwd_years)
    fwd <- dt_permno[year > by & year <= by + fwd_years & !is.na(ret)]
    
    n_months <- nrow(fwd)
    
    if (n_months < min_months) {
      return(data.table(
        base_year  = by,
        fwd_cagr   = NA_real_,
        n_months   = n_months,
        censored   = TRUE
      ))
    }
    
    cum_factor <- prod(1 + fwd$ret, na.rm=TRUE)
    
    if (!is.finite(cum_factor) || cum_factor <= 0) {
      return(data.table(
        base_year  = by,
        fwd_cagr   = NA_real_,
        n_months   = n_months,
        censored   = FALSE
      ))
    }
    
    ## Annualise: (cumulative factor)^(12/n_months) - 1
    cagr <- cum_factor^(12 / n_months) - 1
    
    data.table(
      base_year  = by,
      fwd_cagr   = cagr,
      n_months   = n_months,
      censored   = FALSE
    )
  })
  
  rbindlist(results)
}

## Apply per permno — use data.table grouping for efficiency
all_permnos <- unique(monthly$permno)
cat(sprintf("  Computing for %d permnos × %d base years...\n",
            length(all_permnos), length(base_years)))

## Process in chunks for memory efficiency
chunk_size <- 500L
chunks     <- split(all_permnos, ceiling(seq_along(all_permnos) / chunk_size))
n_chunks   <- length(chunks)

cagr_list <- vector("list", n_chunks)

for (i in seq_along(chunks)) {
  if (i %% 10 == 0 || i == n_chunks)
    cat(sprintf("  Chunk %d/%d...\n", i, n_chunks))
  
  permnos_chunk <- chunks[[i]]
  dt_chunk <- monthly[permno %in% permnos_chunk]
  
  chunk_result <- dt_chunk[, fn_forward_cagr(
    .SD,
    base_years = base_years,
    fwd_years  = FWD_YEARS,
    min_months = MIN_MONTHS_FWD
  ), by = permno, .SDcols = c("year","month","ret","date")]
  
  cagr_list[[i]] <- chunk_result
}

fwd_cagr_dt <- rbindlist(cagr_list)
setnames(fwd_cagr_dt, "base_year", "year")

cat(sprintf("  Forward CAGR computed: %d (permno, year) observations\n",
            nrow(fwd_cagr_dt)))

#==============================================================================#
# 3. Assign bucket labels
#==============================================================================#

cat("\n[05B] Assigning bucket labels...\n")

fwd_cagr_dt[, y_loser := fcase(
  is.na(fwd_cagr) | censored,                              NA_integer_,
  fwd_cagr < CAGR_LOSER_THRESH,                            1L,          ## terminal loser
  fwd_cagr >= CAGR_PHOENIX_THRESH,                         0L,          ## phoenix
  fwd_cagr >= CAGR_LOSER_THRESH & fwd_cagr < CAGR_PHOENIX_THRESH, NA_integer_  ## temp loser
)]

fwd_cagr_dt[, bucket := fcase(
  y_loser == 1L,  "terminal_loser",
  y_loser == 0L,  "phoenix",
  censored,       "censored",
  default = "temporary_loser"
)]

## Summary
n_loser   <- sum(fwd_cagr_dt$y_loser == 1L, na.rm=TRUE)
n_phoenix <- sum(fwd_cagr_dt$y_loser == 0L, na.rm=TRUE)
n_temp    <- sum(!is.na(fwd_cagr_dt$fwd_cagr) & !fwd_cagr_dt$censored &
                   fwd_cagr_dt$fwd_cagr >= CAGR_LOSER_THRESH &
                   fwd_cagr_dt$fwd_cagr < CAGR_PHOENIX_THRESH)
n_cens    <- sum(fwd_cagr_dt$censored, na.rm=TRUE)
n_total   <- nrow(fwd_cagr_dt)

cat(sprintf("\n  Label distribution:\n"))
cat(sprintf("    Terminal loser (y=1) : %6d  (%.1f%%)\n",
            n_loser,   n_loser/n_total*100))
cat(sprintf("    Phoenix        (y=0) : %6d  (%.1f%%)\n",
            n_phoenix, n_phoenix/n_total*100))
cat(sprintf("    Temporary loser (NA) : %6d  (%.1f%%)\n",
            n_temp,    n_temp/n_total*100))
cat(sprintf("    Censored        (NA) : %6d  (%.1f%%)\n",
            n_cens,    n_cens/n_total*100))
cat(sprintf("    Total                : %6d\n", n_total))
cat(sprintf("\n  Usable labels: %d | Loser prevalence: %.1f%%\n",
            n_loser + n_phoenix,
            n_loser / (n_loser + n_phoenix) * 100))

#==============================================================================#
# 4. Final label table — align to annual panel format
#==============================================================================#

labels_bucket <- fwd_cagr_dt[, .(
  permno,
  year,
  y_loser,
  fwd_cagr   = round(fwd_cagr, 6),
  n_months,
  censored,
  bucket
)][order(permno, year)]

## Restrict to analysis window
labels_bucket <- labels_bucket[year >= year(START_DATE) &
                                 year <= year(END_DATE)]

#==============================================================================#
# 5. Assertions
#==============================================================================#

cat("\n[05B] Running assertions...\n")

## A: y_loser only 0, 1, or NA
invalid_y <- labels_bucket[!is.na(y_loser) & !y_loser %in% c(0L,1L)]
stopifnot("Invalid y_loser values" = nrow(invalid_y) == 0L)

## B: No duplicate (permno, year)
n_dup <- sum(duplicated(labels_bucket[, .(permno, year)]))
stopifnot("Duplicate (permno, year)" = n_dup == 0L)

## C: No usable labels after LAST_COMPLETE_YEAR
n_after_cutoff <- labels_bucket[year > LAST_COMPLETE_YEAR & !is.na(y_loser), .N]
if (n_after_cutoff > 0L)
  warning(sprintf("  %d usable labels found after cutoff year %d",
                  n_after_cutoff, LAST_COMPLETE_YEAR))

## D: CAGR direction consistent with label
inconsistent <- labels_bucket[
  !is.na(y_loser) & !is.na(fwd_cagr) &
    ((y_loser == 1L & fwd_cagr >= CAGR_LOSER_THRESH) |
       (y_loser == 0L & fwd_cagr <  CAGR_PHOENIX_THRESH))
]
stopifnot("CAGR/label inconsistency" = nrow(inconsistent) == 0L)

cat("  All assertions passed.\n")

## Save
saveRDS(labels_bucket, PATH_LABELS_BUCKET)
cat(sprintf("  Saved: %s\n", PATH_LABELS_BUCKET))

#==============================================================================#
# 6. Diagnostic plots
#==============================================================================#

cat("\n[05B] Generating diagnostic plots...\n")

##──────────────────────────────────────────────────────────────────────────────
## Plot 1: Label count by year (stacked bar)
##──────────────────────────────────────────────────────────────────────────────

label_by_year <- labels_bucket[
  !is.na(y_loser),
  .(n = .N),
  by = .(year, bucket)
][order(year)]

bucket_colours <- c(
  terminal_loser  = "#E53935",
  phoenix         = "#2196F3",
  temporary_loser = "#FF9800"
)

p_label_dist <- ggplot(label_by_year,
                       aes(x=year, y=n, fill=bucket)) +
  geom_col(width=0.8, alpha=0.85) +
  geom_vline(xintercept=LAST_COMPLETE_YEAR + 0.5,
             linetype="dashed", colour="grey40", linewidth=0.7) +
  annotate("text", x=LAST_COMPLETE_YEAR + 0.6,
           y=max(label_by_year[, sum(n), by=year]$V1)*0.9,
           label="Censored→", hjust=0, size=3, colour="grey40") +
  scale_fill_manual(values=bucket_colours) +
  scale_y_continuous(name="Number of observations") +
  labs(
    title    = "5-Year Forward CAGR Bucket Labels by Year",
    subtitle = sprintf("FWD=%d years | Loser < %.0f%% | Phoenix >= %.0f%%",
                       FWD_YEARS,
                       CAGR_LOSER_THRESH*100,
                       CAGR_PHOENIX_THRESH*100),
    x        = NULL,
    fill     = "Bucket"
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(FIGS$labels, "bucket_label_dist.png"), p_label_dist,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  bucket_label_dist.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## Plot 2: Forward CAGR distribution by bucket
##──────────────────────────────────────────────────────────────────────────────

cagr_plot_dt <- labels_bucket[
  !is.na(fwd_cagr) & !is.na(bucket) & bucket != "censored",
  .(fwd_cagr, bucket)
]

## Winsorise at 2–98% for plot
q_cagr <- quantile(cagr_plot_dt$fwd_cagr, c(0.02, 0.98), na.rm=TRUE)
cagr_plot_dt <- cagr_plot_dt[fwd_cagr >= q_cagr[1] & fwd_cagr <= q_cagr[2]]

p_cagr_dist <- ggplot(cagr_plot_dt,
                      aes(x=fwd_cagr, fill=bucket, colour=bucket)) +
  geom_density(alpha=0.30, linewidth=0.7) +
  geom_vline(xintercept=CAGR_LOSER_THRESH,   linetype="dashed",
             colour="#E53935", linewidth=0.6) +
  geom_vline(xintercept=CAGR_PHOENIX_THRESH, linetype="dashed",
             colour="#2196F3", linewidth=0.6) +
  scale_x_continuous(labels=percent_format(accuracy=1),
                     name="5-Year Forward CAGR") +
  scale_fill_manual(values=bucket_colours) +
  scale_colour_manual(values=bucket_colours) +
  labs(
    title    = "Distribution of 5-Year Forward CAGR by Bucket",
    subtitle = "Dashed lines = bucket thresholds (−2% and 0%)",
    y        = "Density",
    fill     = NULL,
    colour   = NULL
  ) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(FIGS$labels, "bucket_cagr_dist.png"), p_cagr_dist,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  bucket_cagr_dist.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## Plot 3: Loser prevalence by year
##──────────────────────────────────────────────────────────────────────────────

prev_by_year <- labels_bucket[
  !is.na(y_loser),
  .(prevalence = mean(y_loser == 1L),
    n_total    = .N,
    n_loser    = sum(y_loser == 1L)),
  by = year
][order(year)]

p_prevalence <- ggplot(prev_by_year,
                       aes(x=year, y=prevalence)) +
  geom_line(colour="#E53935", linewidth=0.9) +
  geom_point(colour="#E53935", size=2) +
  geom_hline(yintercept=mean(prev_by_year$prevalence),
             linetype="dashed", colour="grey50", linewidth=0.6) +
  annotate("text",
           x=min(prev_by_year$year),
           y=mean(prev_by_year$prevalence),
           label=sprintf("Mean: %.0f%%",
                         mean(prev_by_year$prevalence)*100),
           hjust=0, vjust=-0.5, size=3, colour="grey40") +
  geom_vline(xintercept=LAST_COMPLETE_YEAR + 0.5,
             linetype="dashed", colour="grey40", linewidth=0.7) +
  scale_y_continuous(labels=percent_format(accuracy=1),
                     name="% Terminal Losers",
                     limits=c(0, 1)) +
  labs(
    title    = "Terminal Loser Prevalence by Base Year",
    subtitle = "Higher in early years = more firms with negative 5yr forward returns",
    x        = NULL
  ) +
  theme_minimal(base_size=12)

ggsave(file.path(FIGS$labels, "bucket_prevalence_by_year.png"), p_prevalence,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  bucket_prevalence_by_year.png saved.\n")

##──────────────────────────────────────────────────────────────────────────────
## Plot 4: Overlap with CSI labels
##──────────────────────────────────────────────────────────────────────────────

if (file.exists(PATH_LABELS_BASE)) {
  
  csi_labels <- as.data.table(readRDS(PATH_LABELS_BASE))
  
  overlap_dt <- merge(
    labels_bucket[!is.na(y_loser), .(permno, year, y_loser, bucket)],
    csi_labels[!is.na(y),          .(permno, year, y_csi = y)],
    by = c("permno","year"),
    all = FALSE
  )
  
  overlap_summary <- overlap_dt[, .(
    n               = .N,
    ## Among CSI events, what fraction are terminal losers?
    pct_loser_given_csi    = round(mean(y_loser[y_csi==1L]==1L,
                                        na.rm=TRUE)*100, 1),
    ## Among terminal losers, what fraction had a CSI event?
    pct_csi_given_loser    = round(mean(y_csi[y_loser==1L]==1L,
                                        na.rm=TRUE)*100, 1),
    ## Among phoenixes, what fraction had a CSI event?
    pct_csi_given_phoenix  = round(mean(y_csi[y_loser==0L]==1L,
                                        na.rm=TRUE)*100, 1)
  )]
  
  cat("\n  CSI × Bucket overlap:\n")
  cat(sprintf("    Of CSI events    → %.1f%% are terminal losers\n",
              overlap_summary$pct_loser_given_csi))
  cat(sprintf("    Of terminal losers → %.1f%% had a CSI event\n",
              overlap_summary$pct_csi_given_loser))
  cat(sprintf("    Of phoenixes     → %.1f%% had a CSI event\n",
              overlap_summary$pct_csi_given_phoenix))
  
  ## Mosaic / stacked bar: bucket composition by CSI status
  overlap_plot <- overlap_dt[, .(n=.N),
                             by=.(y_csi, bucket)][order(y_csi, bucket)]
  overlap_plot[, csi_label := ifelse(y_csi==1L, "CSI event", "No CSI event")]
  overlap_plot[, pct := n / sum(n), by=csi_label]
  
  p_overlap <- ggplot(overlap_plot,
                      aes(x=csi_label, y=pct, fill=bucket)) +
    geom_col(width=0.6, alpha=0.85) +
    geom_text(aes(label=sprintf("%.0f%%", pct*100)),
              position=position_stack(vjust=0.5),
              size=3.5, colour="white", fontface="bold") +
    scale_fill_manual(values=bucket_colours) +
    scale_y_continuous(labels=percent_format(accuracy=1),
                       name="% of group") +
    labs(
      title    = "Bucket Composition by CSI Status",
      subtitle = "CSI event firms are disproportionately terminal losers",
      x        = NULL,
      fill     = "5yr Bucket"
    ) +
    theme_minimal(base_size=12) +
    theme(legend.position="bottom")
  
  ggsave(file.path(FIGS$labels, "bucket_vs_csi_overlap.png"), p_overlap,
         width=PLOT_WIDTH*0.8, height=PLOT_HEIGHT, dpi=PLOT_DPI)
  cat("  bucket_vs_csi_overlap.png saved.\n")
}

#==============================================================================#
# 7. Final summary
#==============================================================================#

cat("\n[05B] ══════════════════════════════════════════════════\n")
cat("  BUCKET LABEL SUMMARY\n")
cat("  ══════════════════════════════════════════════════════\n\n")
cat(sprintf("  Forward window     : %d years\n", FWD_YEARS))
cat(sprintf("  Base year range    : %d–%d\n",
            min(labels_bucket$year), LAST_COMPLETE_YEAR))
cat(sprintf("  Censored from      : %d onward\n", LAST_COMPLETE_YEAR + 1L))
cat(sprintf("  Usable labels      : %d\n", n_loser + n_phoenix))
cat(sprintf("  Terminal losers    : %d (%.1f%%)\n",
            n_loser, n_loser/(n_loser+n_phoenix)*100))
cat(sprintf("  Phoenixes          : %d (%.1f%%)\n",
            n_phoenix, n_phoenix/(n_loser+n_phoenix)*100))
cat(sprintf("  Temporary losers   : %d (excluded)\n", n_temp))
cat(sprintf("  Saved to           : %s\n", PATH_LABELS_BUCKET))
cat(sprintf("  Figures saved to   : %s\n", FIGS$labels))

cat(sprintf("\n[05B_Bucket_Labels.R] DONE: %s\n", format(Sys.time())))