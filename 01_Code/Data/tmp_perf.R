source("config.R")
library(data.table)
perf <- readRDS(file.path(DIR_TABLES, "index_performance.rds"))
setDT(perf)

for (per in c("test", "oos")) {
  cat("\n======", toupper(per), "PERIOD ======\n")
  for (wt in c("ew", "mw")) {
    cat("\n--- Weighting:", toupper(wt), "---\n")
    sub <- perf[period == per & weighting == wt &
                  excl_rate %in% c("none","1pct","5pct","10pct","opt")]
    sub[, excl := fifelse(excl_rate == "none", "bench", excl_rate)]
    sub[, short2 := fifelse(is.na(short) | short == "", model_key, short)]
    setorder(sub, track, short2, excl_rate)
    cat(sprintf("  %-12s %-6s %7s %7s %8s %8s %8s\n",
                "Model","Excl","CAGR%","Sharpe","MaxDD%","Sortino","ES97.5%"))
    cat(strrep("-", 72), "\n")
    for (i in seq_len(nrow(sub))) {
      r <- sub[i]
      cat(sprintf("  %-12s %-6s %+6.2f%% %7.3f %+7.2f%% %8.3f %+8.3f%%\n",
                  r$short2, r$excl,
                  r$cagr*100, r$sharpe, r$max_dd*100,
                  ifelse(is.na(r$sortino), NA_real_, r$sortino),
                  r$es_975*100))
    }
  }
}
