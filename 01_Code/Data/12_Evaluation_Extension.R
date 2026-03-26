#==============================================================================#
#==== 12_Evaluation.R — Index Exclusion Diagnostics ==========================#
#==============================================================================#
#
# PURPOSE:
#   Extended diagnostics on the crash-filtered index exclusion decisions.
#   Run after 11_Results.R — loads saved outputs from that script.
#
# SECTIONS:
#   1. Exclusion count over time (firms excluded per strategy per year)
#   2. Performance of excluded firms (how did they actually do?)
#   3. Size of excluded firms (% of total market cap)
#   4. Tracking error vs benchmark
#
# INPUTS:
#   - DIR_TABLES/index_weights.rds
#   - DIR_TABLES/index_exclusion_summary.rds
#   - DIR_TABLES/index_returns.rds
#   - DIR_TABLES/index_performance.rds
#   monthly returns (prices_monthly.rds via PATH_PRICES_MONTHLY)
#   preds_all and ann from 11_Results.R session (or reload below)
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(arrow)
  library(scales)
  library(tidyr)
  library(lubridate)
})

cat("\n[12_Evaluation.R] START:", format(Sys.time()), "\n")

## ── Load saved outputs from 11_Results.R ────────────────────────────────────

weights_all   <- readRDS(file.path(DIR_TABLES, "index_weights.rds"))
excl_summary  <- readRDS(file.path(DIR_TABLES, "index_exclusion_summary.rds"))
port_returns  <- readRDS(file.path(DIR_TABLES, "index_returns.rds"))

## Monthly returns
monthly <- as.data.table(readRDS(PATH_PRICES_MONTHLY))
setnames(monthly, "ret_adj", "ret")
setnames(monthly, "mktcap",  "mkvalt")
monthly[, year  := year(date)]
monthly[, month := month(date)]
monthly[, ret   := pmin(pmax(ret, -0.99, na.rm=TRUE), 10, na.rm=TRUE)]

## Predictions + flags — rebuild ann from saved weights
## Derive excluded firms: in benchmark but not in filtered strategy
bench_ew <- weights_all[strategy == "bench" & weighting == "ew",
                        .(permno, port_year)]

excl_s1 <- weights_all[strategy == "bench" & weighting == "ew",
                       .(permno, port_year)][
                         !weights_all[strategy == "s1" & weighting == "ew",
                                      .(permno, port_year)],
                         on = c("permno","port_year")][, strategy := "s1"]

excl_s2 <- weights_all[strategy == "bench" & weighting == "ew",
                       .(permno, port_year)][
                         !weights_all[strategy == "s2" & weighting == "ew",
                                      .(permno, port_year)],
                         on = c("permno","port_year")][, strategy := "s2"]

excl_s3 <- weights_all[strategy == "bench" & weighting == "ew",
                       .(permno, port_year)][
                         !weights_all[strategy == "s3" & weighting == "ew",
                                      .(permno, port_year)],
                         on = c("permno","port_year")][, strategy := "s3"]

excluded_firms <- rbindlist(list(excl_s1, excl_s2, excl_s3))

## Strategy display labels and colours
STRAT_LABELS <- c(
  bench = "Benchmark",
  s1    = "S1: M1 Ex-Ante",
  s2    = "S2: M3 Triage",
  s3    = "S3: Combined"
)
STRAT_COLOURS <- c(
  bench = "#9E9E9E",
  s1    = "#2196F3",
  s2    = "#F44336",
  s3    = "#4CAF50"
)

OOS_START <- 2016L

#==============================================================================#
# 1. Exclusion count over time
#==============================================================================#

cat("\n[12] Section 1: Exclusion count over time...\n")

## absolute count
excl_count <- excluded_firms[, .(n_excluded = .N), by = .(strategy, port_year)]
setnames(excl_count, "port_year", "year")

## also express as % of universe size that year
universe_size <- excl_summary[, .(year, n_universe)]
excl_count <- merge(excl_count, universe_size, by = "year", all.x = TRUE)
excl_count[, pct_excluded := n_excluded / n_universe]

cat("  Mean exclusions per year:\n")
excl_count[, .(mean_n = round(mean(n_excluded)),
               mean_pct = round(mean(pct_excluded)*100, 1)),
           by = strategy] |> print()

## Plot 1A: absolute count
p_excl_abs <- ggplot(excl_count,
                     aes(x=year, y=n_excluded,
                         colour=strategy, group=strategy)) +
  geom_line(linewidth=0.9) +
  geom_point(size=2, alpha=0.7) +
  geom_vline(xintercept=OOS_START - 0.5,
             linetype="dashed", colour="grey40", linewidth=0.7) +
  annotate("text", x=OOS_START - 0.3, y=max(excl_count$n_excluded)*0.95,
           label="OOS→", hjust=0, size=3, colour="grey40") +
  scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  scale_y_continuous(name="Firms excluded") +
  labs(title="Number of Firms Excluded per Year",
       subtitle="Rank-based exclusion (top 5% by predicted CSI score per year)",
       x=NULL, colour="Strategy") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES, "diag_exclusion_count_abs.png"), p_excl_abs,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  diag_exclusion_count_abs.png saved.\n")

## Plot 1B: % of universe
p_excl_pct <- ggplot(excl_count,
                     aes(x=year, y=pct_excluded,
                         colour=strategy, group=strategy)) +
  geom_line(linewidth=0.9) +
  geom_point(size=2, alpha=0.7) +
  geom_vline(xintercept=OOS_START - 0.5,
             linetype="dashed", colour="grey40", linewidth=0.7) +
  scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  scale_y_continuous(labels=percent_format(accuracy=0.1),
                     name="% of universe excluded") +
  labs(title="Exclusion Rate as % of Universe per Year",
       subtitle="Should be close to 5% for S1/S2 (rank-based), ~10% for S3 (union)",
       x=NULL, colour="Strategy") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES, "diag_exclusion_pct.png"), p_excl_pct,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  diag_exclusion_pct.png saved.\n")

#==============================================================================#
# 2. Performance of excluded firms
#==============================================================================#

cat("\n[12] Section 2: Performance of excluded firms...\n")

## For each excluded firm, compute its return over the year it was excluded
## port_year = the year the firm is held (or excluded) → use monthly returns
##             for that calendar year

excl_annual_ret <- merge(
  excluded_firms,
  monthly[, .(permno, year, ret)],
  by.x = c("permno", "port_year"),
  by.y = c("permno", "year"),
  all.x = FALSE
)

## Annual return per excluded firm
excl_annual_ret <- excl_annual_ret[, .(
  ann_ret = prod(1 + ret, na.rm=TRUE) - 1,
  n_months = .N
), by = .(permno, port_year, strategy)]

## Also get the included firms (benchmark) for comparison
incl_annual_ret <- merge(
  bench_ew,
  monthly[, .(permno, year, ret)],
  by.x = c("permno", "port_year"),
  by.y = c("permno", "year"),
  all.x = FALSE
)[, .(
  ann_ret  = prod(1 + ret, na.rm=TRUE) - 1,
  n_months = .N
), by = .(permno, port_year)]
incl_annual_ret[, strategy := "bench"]

## Summary statistics
cat("\n  Annual return distribution — excluded vs included firms:\n")
cat("  (negative mean for excluded = good; strategy correctly excluded losers)\n\n")

ret_summary <- rbindlist(list(excl_annual_ret, incl_annual_ret))[
  n_months >= 6  ## require at least 6 months of data
][, .(
  n_firms   = .N,
  mean_ret  = round(mean(ann_ret,   na.rm=TRUE)*100, 2),
  median_ret = round(median(ann_ret, na.rm=TRUE)*100, 2),
  sd_ret    = round(sd(ann_ret,     na.rm=TRUE)*100, 2),
  pct_neg   = round(mean(ann_ret < 0, na.rm=TRUE)*100, 1),
  pct_below_minus20 = round(mean(ann_ret < -0.20, na.rm=TRUE)*100, 1),
  pct_below_minus50 = round(mean(ann_ret < -0.50, na.rm=TRUE)*100, 1)
), by = strategy]

print(ret_summary, row.names = FALSE)

## Plot 2A: density of annual returns — excluded S1 + S2 vs benchmark included
plot_dt <- rbindlist(list(
  excl_annual_ret[strategy %in% c("s1","s2") & n_months >= 6,
                  .(ann_ret, group = paste0(STRAT_LABELS[strategy], " (excluded)"))],
  incl_annual_ret[n_months >= 6,
                  .(ann_ret, group = "Benchmark (included)")]
))

p_excl_dist <- ggplot(plot_dt[ann_ret >= -1 & ann_ret <= 3],
                      aes(x=ann_ret, fill=group, colour=group)) +
  geom_density(alpha=0.25, linewidth=0.7) +
  geom_vline(xintercept=0, linetype="dashed", colour="grey40") +
  scale_x_continuous(labels=percent_format(accuracy=1),
                     name="Annual Return (year of exclusion)") +
  scale_fill_manual(values=c(
    "S1: M1 Ex-Ante (excluded)" = "#2196F3",
    "S2: M3 Triage (excluded)"  = "#F44336",
    "Benchmark (included)"      = "#9E9E9E"
  )) +
  scale_colour_manual(values=c(
    "S1: M1 Ex-Ante (excluded)" = "#2196F3",
    "S2: M3 Triage (excluded)"  = "#F44336",
    "Benchmark (included)"      = "#9E9E9E"
  )) +
  labs(title="Return Distribution: Excluded vs Included Firms",
       subtitle="Year of exclusion | Left shift = model correctly excludes underperformers",
       y="Density", fill=NULL, colour=NULL) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES, "diag_excluded_return_dist.png"), p_excl_dist,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  diag_excluded_return_dist.png saved.\n")

## Plot 2B: mean return of excluded firms by year
excl_mean_ret <- excl_annual_ret[n_months >= 6,
                                 .(mean_ret = mean(ann_ret, na.rm=TRUE)),
                                 by = .(strategy, port_year)]
setnames(excl_mean_ret, "port_year", "year")

bench_mean_ret <- incl_annual_ret[n_months >= 6,
                                  .(mean_ret = mean(ann_ret, na.rm=TRUE),
                                    strategy = "bench"),
                                  by = .(year = port_year)]

ret_by_year <- rbindlist(list(excl_mean_ret[strategy %in% c("s1","s2")],
                              bench_mean_ret))

p_excl_ret_yr <- ggplot(ret_by_year,
                        aes(x=year, y=mean_ret,
                            colour=strategy, group=strategy)) +
  geom_line(linewidth=0.9) +
  geom_hline(yintercept=0, colour="black", linewidth=0.3) +
  geom_vline(xintercept=OOS_START - 0.5,
             linetype="dashed", colour="grey40", linewidth=0.7) +
  scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  scale_y_continuous(labels=percent_format(accuracy=1),
                     name="Mean Annual Return") +
  labs(title="Mean Return of Excluded Firms vs Benchmark Included",
       subtitle="Excluded below benchmark line = model adds value by excluding them",
       x=NULL, colour=NULL) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES, "diag_excluded_return_by_year.png"), p_excl_ret_yr,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  diag_excluded_return_by_year.png saved.\n")

#==============================================================================#
# 3. Size of excluded firms (% of total market cap)
#==============================================================================#

cat("\n[12] Section 3: Size of excluded firms...\n")

## Get market cap for each excluded firm at time of exclusion
## Use December mktcap from monthly data (port_year - 1 = the ranking year)
dec_mktcap <- monthly[month == 12L & !is.na(mkvalt),
                      .(mkvalt_dec = last(mkvalt)),
                      by = .(permno, year)]

## For excluded firms: their mktcap at year of prediction (port_year - 1)
excl_mktcap <- merge(
  excluded_firms[, .(permno, port_year, strategy)],
  dec_mktcap[, .(permno, year, mkvalt_dec)],
  by.x = c("permno", "port_year"),
  by.y = c("permno", "year"),
  all.x = TRUE
)
## Note: port_year is the holding year, so the prediction/ranking was in
## port_year - 1. Merge on port_year directly since we want the cap at
## the time of rebalancing decision (end of port_year - 1 = start of port_year)

## Total market cap of the full universe each year
total_mktcap <- merge(
  bench_ew,
  dec_mktcap,
  by.x = c("permno", "port_year"),
  by.y = c("permno", "year"),
  all.x = FALSE
)[, .(total_mktcap = sum(mkvalt_dec, na.rm=TRUE)), by = port_year]

## Market cap of excluded firms
excl_mktcap_yr <- excl_mktcap[!is.na(mkvalt_dec),
                              .(excl_mktcap = sum(mkvalt_dec, na.rm=TRUE)),
                              by = .(strategy, port_year)]

size_dt <- merge(excl_mktcap_yr, total_mktcap,
                 by = "port_year", all.x = TRUE)
size_dt[, pct_mktcap := excl_mktcap / total_mktcap]

cat("\n  Excluded firms as % of total universe market cap:\n")
size_dt[, .(mean_pct = round(mean(pct_mktcap)*100, 2)),
        by = strategy] |> print()

## Plot 3A: % of total mktcap excluded over time
p_size_pct <- ggplot(size_dt,
                     aes(x=port_year, y=pct_mktcap,
                         colour=strategy, group=strategy)) +
  geom_line(linewidth=0.9) +
  geom_point(size=2, alpha=0.7) +
  geom_vline(xintercept=OOS_START - 0.5,
             linetype="dashed", colour="grey40", linewidth=0.7) +
  scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  scale_y_continuous(labels=percent_format(accuracy=0.1),
                     name="Excluded firms as % of total universe mktcap") +
  labs(title="Market Cap Weight of Excluded Firms",
       subtitle="Low % = excluded firms are small-caps (minimal cap-weight impact)",
       x=NULL, colour="Strategy") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES, "diag_excluded_mktcap_pct.png"), p_size_pct,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  diag_excluded_mktcap_pct.png saved.\n")

## Plot 3B: size distribution of excluded vs included firms (mktcap histogram)
## Use log scale — mktcap spans many orders of magnitude
excl_size_dist <- merge(
  excluded_firms[strategy %in% c("s1","s2")],
  dec_mktcap,
  by.x = c("permno","port_year"),
  by.y = c("permno","year"),
  all.x = FALSE
)[, group := paste0(STRAT_LABELS[strategy], " (excluded)")]

incl_size_dist <- merge(
  bench_ew,
  dec_mktcap,
  by.x = c("permno","port_year"),
  by.y = c("permno","year"),
  all.x = FALSE
)[, group := "Benchmark (included)"]

size_dist_dt <- rbindlist(list(
  excl_size_dist[, .(mkvalt_dec, group)],
  incl_size_dist[, .(mkvalt_dec, group)]
))[mkvalt_dec > 0]

p_size_dist <- ggplot(size_dist_dt,
                      aes(x=mkvalt_dec/1e3, fill=group, colour=group)) +
  geom_density(alpha=0.25, linewidth=0.7) +
  scale_x_log10(labels=dollar_format(suffix="B", scale=1e-3,
                                     accuracy=0.1),
                name="Market Cap (log scale, $B)") +
  scale_fill_manual(values=c(
    "S1: M1 Ex-Ante (excluded)" = "#2196F3",
    "S2: M3 Triage (excluded)"  = "#F44336",
    "Benchmark (included)"      = "#9E9E9E"
  )) +
  scale_colour_manual(values=c(
    "S1: M1 Ex-Ante (excluded)" = "#2196F3",
    "S2: M3 Triage (excluded)"  = "#F44336",
    "Benchmark (included)"      = "#9E9E9E"
  )) +
  labs(title="Size Distribution: Excluded vs Included Firms",
       subtitle="Left shift = excluded firms are smaller than average",
       y="Density", fill=NULL, colour=NULL) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom")

ggsave(file.path(DIR_FIGURES, "diag_excluded_size_dist.png"), p_size_dist,
       width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  diag_excluded_size_dist.png saved.\n")

#==============================================================================#
# 4. Tracking error vs benchmark
#==============================================================================#

cat("\n[12] Section 4: Tracking error vs benchmark...\n")

## Tracking error = annualised standard deviation of (strategy return - benchmark return)
bench_ret <- port_returns[strategy == "bench" & weighting == "ew",
                          .(date, year, month, bench_ret = port_ret)]

te_rows <- list()

for (strat in c("s1","s2","s3")) {
  for (wt in c("ew","cw")) {
    
    strat_ret <- port_returns[strategy == strat & weighting == wt,
                              .(date, port_ret)]
    
    ## Use equal-weight benchmark for EW strategies, cap-weight for CW
    if (wt == "cw") {
      bm <- port_returns[strategy == "bench" & weighting == "cw",
                         .(date, bench_ret = port_ret)]
    } else {
      bm <- bench_ret[, .(date, bench_ret)]
    }
    
    merged <- merge(strat_ret, bm, by="date", all=FALSE)
    merged[, excess := port_ret - bench_ret]
    merged[, year   := year(date)]
    
    ## Full period TE
    te_full <- sd(merged$excess, na.rm=TRUE) * sqrt(12)
    
    ## Rolling 12-month TE
    merged[, roll_te := zoo::rollapply(excess, 12,
                                       function(x) sd(x)*sqrt(12),
                                       fill=NA, align="right")]
    
    te_rows[[length(te_rows)+1]] <- data.frame(
      strategy  = strat,
      weighting = wt,
      te_annual = round(te_full * 100, 3),
      stringsAsFactors = FALSE
    )
    
    ## Save rolling TE for plot
    merged[, strategy  := strat]
    merged[, weighting := wt]
    if (strat == "s1" && wt == "ew") roll_te_all <- merged
    else roll_te_all <- rbindlist(list(roll_te_all, merged))
  }
}

te_table <- do.call(rbind, te_rows)

cat("\n  Annualised tracking error (equal-weight):\n")
print(te_table[te_table$weighting == "ew", ], row.names = FALSE)

cat("\n  Annualised tracking error (cap-weight):\n")
print(te_table[te_table$weighting == "cw", ], row.names = FALSE)

## Plot 4A: Rolling 12-month tracking error — EW strategies
roll_te_ew <- roll_te_all[weighting == "ew" & !is.na(roll_te)]

p_te_roll <- ggplot(roll_te_ew,
                    aes(x=date, y=roll_te,
                        colour=strategy, group=strategy)) +
  geom_line(linewidth=0.9) +
  geom_vline(xintercept=as.numeric(as.Date(sprintf("%d-01-01", OOS_START))),
             linetype="dashed", colour="grey40", linewidth=0.7) +
  annotate("text",
           x=as.Date(sprintf("%d-01-01", OOS_START)),
           y=max(roll_te_ew$roll_te, na.rm=TRUE)*0.95,
           label="OOS→", hjust=-0.1, size=3, colour="grey40") +
  scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  scale_y_continuous(labels=percent_format(accuracy=0.1),
                     name="Rolling 12m Tracking Error (annualised)") +
  scale_x_date(date_breaks="2 years", date_labels="%Y") +
  labs(title="Rolling 12-Month Tracking Error vs Equal-Weight Benchmark",
       subtitle="Higher TE = strategy diverges more from benchmark",
       x=NULL, colour="Strategy") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",
        axis.text.x=element_text(angle=30, hjust=1))

ggsave(file.path(DIR_FIGURES, "diag_tracking_error_rolling.png"), p_te_roll,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  diag_tracking_error_rolling.png saved.\n")

## Plot 4B: Active return (strategy - benchmark) over time
active_ret_ew <- roll_te_all[weighting == "ew",
                             .(date, strategy, excess, year = year(date))]
active_ret_ew[, cum_active := cumsum(excess), by = strategy]

p_active <- ggplot(active_ret_ew,
                   aes(x=date, y=cum_active,
                       colour=strategy, group=strategy)) +
  geom_line(linewidth=0.9) +
  geom_hline(yintercept=0, colour="black", linewidth=0.4) +
  geom_vline(xintercept=as.numeric(as.Date(sprintf("%d-01-01", OOS_START))),
             linetype="dashed", colour="grey40", linewidth=0.7) +
  scale_colour_manual(values=STRAT_COLOURS, labels=STRAT_LABELS) +
  scale_y_continuous(labels=percent_format(accuracy=1),
                     name="Cumulative Active Return vs Benchmark") +
  scale_x_date(date_breaks="2 years", date_labels="%Y") +
  labs(title="Cumulative Active Return vs Equal-Weight Benchmark",
       subtitle="Above zero = outperformance | Below zero = underperformance",
       x=NULL, colour="Strategy") +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",
        axis.text.x=element_text(angle=30, hjust=1))

ggsave(file.path(DIR_FIGURES, "diag_active_return.png"), p_active,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  diag_active_return.png saved.\n")

#==============================================================================#
# 5. Console summary
#==============================================================================#

cat("\n[12] ══════════════════════════════════════════════════════\n")
cat("  EXCLUSION DIAGNOSTICS SUMMARY\n")
cat("  ══════════════════════════════════════════════════════\n\n")

cat("  1. EXCLUSION COUNT\n")
excl_count[, .(
  strategy,
  mean_n_excluded = round(mean(n_excluded)),
  mean_pct = paste0(round(mean(pct_excluded)*100, 1), "%")
), by = strategy][!duplicated(strategy)] |>
  print(row.names=FALSE)

cat("\n  2. EXCLUDED FIRM RETURNS (mean annual, full period)\n")
print(ret_summary[, .(strategy, mean_ret, median_ret,
                      pct_neg, pct_below_minus20)],
      row.names=FALSE)

cat("\n  3. EXCLUDED FIRM SIZE (% of universe mktcap)\n")
size_dt[, .(mean_pct_mktcap = paste0(round(mean(pct_mktcap)*100,2),"%")),
        by=strategy] |> print(row.names=FALSE)

cat("\n  4. TRACKING ERROR (annualised, EW)\n")
print(te_table[te_table$weighting=="ew",
               c("strategy","te_annual")], row.names=FALSE)

cat(sprintf("\n[12_Evaluation.R] DONE: %s\n", format(Sys.time())))