## tmp_perf_quality.R
## Full performance table matching Doc2 Performance sheet structure.
## C1 strategies shown at "opt" exclusion rate (best F1-threshold from CV).

source("config.R")
library(data.table)

perf <- readRDS(file.path(DIR_TABLES, "quality_performance.rds"))
setDT(perf)

PERIODS_ORDER <- c("insample", "test", "oos", "full")
PERIOD_LABELS <- c(insample = "Train 1993-2015",
                   test     = "Test  2016-2019",
                   oos      = "OOS   2020-2024",
                   full     = "Full  1993-2024")

## Rows to display (model_key / excl_rate / weighting / display label)
ROWS <- list(
  ## Benchmarks
  list(mk="bench",       er="none", wt="mw", lbl="Benchmark MW (Full Universe)"),
  list(mk="bench",       er="none", wt="ew", lbl="Benchmark EW (Full Universe)"),
  ## USMV proxies
  list(mk="usmv_proxy",  er="none", wt="mw", lbl="USMV Proxy MW (Bottom Quintile Vol)"),
  list(mk="usmv_proxy",  er="none", wt="ew", lbl="USMV Proxy EW (Bottom Quintile Vol)"),
  ## QUAL proxies
  list(mk="qual_proxy",  er="none", wt="mw", lbl="QUAL Proxy MW (Top Quintile Quality)"),
  list(mk="qual_proxy",  er="none", wt="ew", lbl="QUAL Proxy EW (Top Quintile Quality)"),
  ## C1/C3 — best CSI (M3-Raw, opt threshold)
  list(mk="c1_csi",      er="opt",  wt="mw", lbl="C1 MW | Best CSI (M3-Raw, Opt)"),
  list(mk="c1_csi",      er="opt",  wt="ew", lbl="C3 EW | Best CSI (M3-Raw, Opt)"),
  ## C1/C3 — best Bucket (B1-Fund, opt threshold)
  list(mk="c1_bucket",   er="opt",  wt="mw", lbl="C1 MW | Best Bucket (B1-Fund, Opt)"),
  list(mk="c1_bucket",   er="opt",  wt="ew", lbl="C3 EW | Best Bucket (B1-Fund, Opt)"),
  ## C1/C3 — best Combined (B1 union Struct-Raw, opt threshold)
  list(mk="c1_comb",     er="opt",  wt="mw", lbl="C1 MW | Best Combined (B1 u S-Raw, Opt)"),
  list(mk="c1_comb",     er="opt",  wt="ew", lbl="C3 EW | Best Combined (B1 u S-Raw, Opt)")
)

## Helper: pull one row from perf
pull <- function(mk, er, wt, per_nm) {
  perf[model_key == mk & excl_rate == er & weighting == wt & period == per_nm]
}

fmt_pct  <- function(x) if (is.na(x) || !is.finite(x)) "   —   " else sprintf("%+7.2f%%", x * 100)
fmt_x    <- function(x) if (is.na(x) || !is.finite(x)) "   —   " else sprintf("%7.3f",    x)

cat("\n")
cat(strrep("=", 115), "\n")
cat(sprintf("%-42s  %9s %9s %9s %8s %8s %8s %8s\n",
            "Strategy",
            "Ann.Ret%", "Vol%", "Sharpe", "MaxDD%", "Calmar", "ES97.5%", "ES99%"))
cat(strrep("=", 115), "\n")

## Collect all numbers for CSV export
csv_rows <- list()

for (per in PERIODS_ORDER) {
  cat(sprintf("\n  ── %s ──\n", PERIOD_LABELS[per]))
  cat(strrep("-", 115), "\n")

  ## Grab QUAL proxy MW for delta computation
  q_mw <- pull("qual_proxy", "none", "mw", per)
  u_mw <- pull("usmv_proxy", "none", "mw", per)

  for (r in ROWS) {
    d <- pull(r$mk, r$er, r$wt, per)
    if (nrow(d) > 1L) d <- d[1L]
    if (nrow(d) == 0) {
      cat(sprintf("  %-42s  [no data]\n", r$lbl))
      next
    }
    cat(sprintf("  %-42s  %s %s %s %s %s %s %s\n",
                r$lbl,
                fmt_pct(d$cagr), fmt_pct(d$vol),
                fmt_x(d$sharpe), fmt_pct(d$max_dd),
                fmt_x(d$calmar), fmt_pct(d$es_975), fmt_pct(d$es_99)))

    csv_rows[[length(csv_rows)+1]] <- data.table(
      period=per, label=r$lbl, model_key=r$mk, excl_rate=r$er, weighting=r$wt,
      cagr=d$cagr, vol=d$vol, sharpe=d$sharpe, max_dd=d$max_dd,
      calmar=d$calmar, es_975=d$es_975, es_99=d$es_99
    )
  }

  ## Deltas: C1 (opt, mw) vs QUAL proxy MW
  if (nrow(q_mw) > 0) {
    cat(strrep("-", 115), "\n")
    cat(sprintf("  %s\n", "DELTA vs. QUAL Proxy MW:"))
    for (delta_mk in c("c1_csi", "c1_bucket", "c1_comb")) {
      dm <- pull(delta_mk, "opt", "mw", per)
      if (nrow(dm) == 0 || nrow(q_mw) == 0) next
      dlbl <- switch(delta_mk,
                     c1_csi    = "  Δ C1-CSI    vs QUAL Proxy MW",
                     c1_bucket = "  Δ C1-Bucket vs QUAL Proxy MW",
                     c1_comb   = "  Δ C1-Comb   vs QUAL Proxy MW")
      cat(sprintf("  %-42s  %s %s %s %s %s %s %s\n",
                  dlbl,
                  fmt_pct(dm$cagr    - q_mw$cagr),
                  fmt_pct(dm$vol     - q_mw$vol),
                  fmt_x(  dm$sharpe  - q_mw$sharpe),
                  fmt_pct(dm$max_dd  - q_mw$max_dd),
                  fmt_x(  dm$calmar  - q_mw$calmar),
                  fmt_pct(dm$es_975  - q_mw$es_975),
                  fmt_pct(dm$es_99   - q_mw$es_99)))
    }
  }

  ## Deltas vs USMV proxy MW (MaxDD + ES focus)
  if (nrow(u_mw) > 0) {
    cat(sprintf("  %s\n", "DELTA vs. USMV Proxy MW (risk focus):"))
    for (delta_mk in c("c1_csi", "c1_bucket", "c1_comb")) {
      dm <- pull(delta_mk, "opt", "mw", per)
      if (nrow(dm) == 0) next
      dlbl <- switch(delta_mk,
                     c1_csi    = "  Δ C1-CSI    vs USMV Proxy MW",
                     c1_bucket = "  Δ C1-Bucket vs USMV Proxy MW",
                     c1_comb   = "  Δ C1-Comb   vs USMV Proxy MW")
      cat(sprintf("  %-42s  %s %s %s %s %s %s %s\n",
                  dlbl,
                  fmt_pct(dm$cagr    - u_mw$cagr),
                  fmt_pct(dm$vol     - u_mw$vol),
                  fmt_x(  dm$sharpe  - u_mw$sharpe),
                  fmt_pct(dm$max_dd  - u_mw$max_dd),
                  fmt_x(  dm$calmar  - u_mw$calmar),
                  fmt_pct(dm$es_975  - u_mw$es_975),
                  fmt_pct(dm$es_99   - u_mw$es_99)))
    }
  }
}

cat("\n", strrep("=", 115), "\n")

## Save CSV
csv_out <- rbindlist(csv_rows, fill=TRUE)
write.csv(csv_out, file.path(DIR_TABLES, "quality_performance_table.csv"),
          row.names=FALSE)
cat(sprintf("\nCSV saved: %s\n", file.path(DIR_TABLES, "quality_performance_table.csv")))
