#==============================================================================#
#==== 05_CSI_Label.R ==========================================================#
#==== Catastrophic Stock Implosion — Label Construction =======================#
#==============================================================================#
#
# PURPOSE:
#   Construct binary CSI labels at (permno, year) granularity for use as
#   the dependent variable in the machine learning pipeline.
#
# DEFINITION:
#   A Catastrophic Stock Implosion (CSI) occurs when:
#     1. The drawdown from the rolling peak crosses PARAM_C in month t
#        (stock has lost >= |PARAM_C| from its all-time high to date)
#     2. From the crash low, the stock does not recover more than
#        abs(PARAM_M) within the following PARAM_T months
#        (zombie condition confirmed retrospectively)
#
#   DEVIATIONS FROM PAPER (Tewari, Galas, Treleaven):
#     - Frequency   : Monthly prices (paper uses weekly). Defensible given
#                     annual label horizon and Compustat's quarterly reporting.
#     - Drawdown    : Peak-to-trough wealth index (paper uses cumulative return
#                     from unspecified baseline — our definition is cleaner).
#     - Multi-event : Multiple CSI events allowed per stock (paper: one per stock).
#                     Rationale: firms can recover and crash again; more labels
#                     improves class balance and reflects real portfolio dynamics.
#     - Sensitivity : Full 27-combination grid (paper uses single parameterisation).
#     - T unit      : MONTHS throughout (config.R CSI_BASE$T = 18 months ≈ 78 weeks).
#
# LABEL SCHEMA:
#   y(s, t) = 1   stock s has confirmed CSI trigger in calendar year t
#   y(s, t) = 0   stock s alive in year t, no CSI trigger fires
#   y(s, t) = NA  zombie period: t falls within PARAM_T months post-event
#   y(s, t) = NA  censored: trigger fires but t + PARAM_T > END_DATE
#
# LABEL PRIORITY (monthly → annual collapse):
#   1 > NA > 0
#   One confirmed crash month in year t → y(t) = 1
#   Any zombie/censored month without crash → y(t) = NA
#   All clean months → y(t) = 0
#
# INPUTS:
#   - config.R
#   - PATH_PRICES_MONTHLY  : permno, date, ret_adj
#   - PATH_UNIVERSE        : permno list
#
# OUTPUTS:
#   - PATH_LABELS_BASE     : base case labels (permno, year, y, censored)
#   - PATH_LABELS_GRID     : all 27 grid results stacked with param_id
#   - PATH_LABELS_DIAG     : prevalence and event counts per grid combination
#   - PATH_FIGURE_CSI      : bar chart of CSI events per year (base case)
#   - DIR_LABELS/labels_<param_id>.rds : one file per CSI_GRID row (27 total)
#
#==============================================================================#

source("config.R")

cat("\n[05_CSI_Label.R] START:", format(Sys.time()), "\n")

#==============================================================================#
# 0. Load inputs
#==============================================================================#

## prices_weekly intentionally NOT loaded — label pipeline runs on monthly prices.
## Weekly output (PATH_PRICES_WEEKLY) is used by feature engineering only.
prices_monthly <- readRDS(PATH_PRICES_MONTHLY)
universe       <- readRDS(PATH_UNIVERSE)

## Restrict to universe permno and sort
prices_monthly <- prices_monthly |>
  filter(permno %in% universe$permno) |>
  arrange(permno, date)

cat("[05_CSI_Label.R] Loaded:", n_distinct(prices_monthly$permno),
    "permno,", nrow(prices_monthly), "monthly observations\n")

#==============================================================================#
# 1. Data quality assertion — consecutive return gaps
#
#    MAX_CONSECUTIVE_NA is defined in config.R.
#    If any stock exceeds this threshold the pipeline halts loudly —
#    the drawdown and wealth index computation would be unreliable.
#==============================================================================#

cat("[05_CSI_Label.R] Checking return data quality...\n")

dt_gap <- as.data.table(prices_monthly)[, .(permno, date, ret_adj)]
dt_gap[, is_na     := is.na(ret_adj)]
dt_gap[, consec_na := if_else(
  is_na,
  rowid(rleid(is_na)),   # Resets to 1 at start of each new NA run
  0L
), by = permno]

gap_check <- dt_gap[
  , .(max_consec_na = max(consec_na)), by = permno
][max_consec_na >= MAX_CONSECUTIVE_NA]

if (nrow(gap_check) > 0) {
  cat("[05_CSI_Label.R] Permno with consecutive NA returns >= threshold:\n")
  print(gap_check)
  stop(sprintf(
    "[05_CSI_Label.R] ASSERTION FAILED: %d stocks have >= %d consecutive missing returns.",
    nrow(gap_check), MAX_CONSECUTIVE_NA
  ))
}

cat("[05_CSI_Label.R] Data quality check passed.\n")

#==============================================================================#
# 2. Core functions
#==============================================================================#

##----------------------------------------------------------------------------##
## fn_detect_csi_events
##
## Identifies confirmed CSI events for a single stock given parameters C, M, T.
##
## Called via data.table by = permno. Receives .SD — a subset of the full
## data.table for one permno, containing columns:
##   date, ret_adj, wealth_index, running_peak, drawdown
##
## NOTE: permno is the grouping key and is NOT present in .SD. The by = permno
##       in the calling expression handles grouping — do not reference permno here.
##
## LOGIC:
##   Scan forward month by month.
##   When drawdown < C (crash trigger):
##     - crash_low        = wealth_index at trigger month
##     - recovery_ceiling = crash_low * (1 + abs(M))
##     - Look forward T months
##     - If max(wealth_index) in [trigger+1, trigger+T] <= recovery_ceiling:
##         → zombie confirmed → mark trigger + zombie window
##         → resume scanning after zombie window (allows multiple events)
##     - If trigger + T > END_DATE:
##         → censored → unknown outcome → mark as censored
##     - If max > recovery_ceiling:
##         → stock recovered → not CSI → continue scanning
##
## INPUT:  .SD data.table for one permno, sorted by date
##         C, M, T : scalar parameters (T in months)
##         end_date: Date scalar for right-censoring cutoff
##
## OUTPUT: .SD with added column csi_status:
##           "trigger"  — confirmed CSI month
##           "zombie"   — post-event zombie window (excluded from labels)
##           "censored" — trigger fired but zombie unconfirmable (y = NA)
##           "normal"   — clean observation (y = 0)
##----------------------------------------------------------------------------##

fn_detect_csi_events <- function(dt, C, M, T, end_date) {
  
  n      <- nrow(dt)
  status <- rep("normal", n)
  
  i <- 1L
  while (i <= n) {
    
    ## Skip months already marked (inside a previous zombie window)
    if (status[i] != "normal") {
      i <- i + 1L
      next
    }
    
    ## Check crash trigger
    if (!is.na(dt$drawdown[i]) && dt$drawdown[i] < C) {
      
      crash_low        <- dt$wealth_index[i]
      recovery_ceiling <- crash_low * (1 + abs(M))
      trigger_date     <- dt$date[i]
      zombie_end_date  <- trigger_date %m+% months(T)   # T is in MONTHS
      
      ## Right-censoring: zombie window extends beyond END_DATE
      if (zombie_end_date > end_date) {
        status[i] <- "censored"
        i <- i + 1L
        next
      }
      
      ## Identify forward index range for zombie window
      zombie_end_idx <- min(i + T, n)
      forward_idx    <- seq(i + 1L, zombie_end_idx)
      
      if (length(forward_idx) == 0L) {
        ## No forward observations — treat as censored
        status[i] <- "censored"
        i <- i + 1L
        next
      }
      
      forward_max <- max(dt$wealth_index[forward_idx], na.rm = TRUE)
      
      if (forward_max <= recovery_ceiling) {
        ## Zombie confirmed — CSI event
        status[i]           <- "trigger"
        status[forward_idx] <- "zombie"
        
        ## Resume scanning after zombie window
        i <- max(forward_idx) + 1L
        
      } else {
        ## Stock recovered — not a CSI event
        i <- i + 1L
      }
      
    } else {
      i <- i + 1L
    }
  }
  
  return(data.table(
    date         = dt$date,
    ret_adj      = dt$ret_adj,
    wealth_index = dt$wealth_index,
    running_peak = dt$running_peak,
    drawdown     = dt$drawdown,
    csi_status   = status
  ))
}

##----------------------------------------------------------------------------##
## fn_build_annual_panel
##
## Collapses monthly CSI status to annual (permno, year) labels.
##
## PRIORITY RULE within each year:
##   "trigger"  → y = 1   (confirmed CSI — highest priority)
##   "censored" → y = NA  (unknown outcome)
##   "zombie"   → y = NA  (post-event exclusion)
##   "normal"   → y = 0   (clean — only if no other status present)
##
## One confirmed crash month makes the entire year y = 1.
## Any ambiguous month without a trigger makes the year y = NA.
##----------------------------------------------------------------------------##

fn_build_annual_panel <- function(dt_monthly_labelled) {
  
  dt <- copy(dt_monthly_labelled)
  dt[, year := year(date)]
  
  annual <- dt[, .(
    has_trigger   = any(csi_status == "trigger"),
    has_ambiguous = any(csi_status %in% c("censored", "zombie"))
  ), by = .(permno, year)]
  
  annual[, y := fcase(
    has_trigger,                   1L,
    !has_trigger & has_ambiguous,  NA_integer_,
    !has_trigger & !has_ambiguous, 0L
  )]
  
  annual[, censored := (!has_trigger & has_ambiguous)]
  annual[, c("has_trigger", "has_ambiguous") := NULL]
  
  return(annual[, .(permno, year, y, censored)])
}

#==============================================================================#
# 3. Master label construction function
#
#    Wraps drawdown computation + event detection + annual collapse
#    for one parameter combination. Called once per CSI_GRID row.
#==============================================================================#

fn_run_csi_pipeline <- function(prices_dt, C, M, T, end_date, param_id) {
  
  cat(sprintf("  [%s] C=%.2f | M=%.2f | T=%d months\n", param_id, C, M, T))
  
  ##--------------------------------------------------------------------------##
  ## Stage 1: Drawdown computation inline by permno
  ##
  ##   ret_safe fills NA returns with 0 ONLY after the quality assertion
  ##   above has confirmed no problematic gaps exist. Isolated NAs (e.g.,
  ##   first month of listing) are handled safely — a single 0-fill does
  ##   not meaningfully distort the wealth index.
  ##--------------------------------------------------------------------------##
  
  prices_dt <- prices_dt[order(permno, date)]
  prices_dt[, ret_safe     := fifelse(is.na(ret_adj), 0, ret_adj)]
  prices_dt[, wealth_index := cumprod(1 + ret_safe),  by = permno]
  prices_dt[, running_peak := cummax(wealth_index),   by = permno]
  prices_dt[, drawdown     := wealth_index / running_peak - 1, by = permno]
  prices_dt[, ret_safe     := NULL]
  
  ##--------------------------------------------------------------------------##
  ## Stage 2: CSI event detection by permno
  ##
  ##   .SDcols explicitly lists only the columns fn_detect_csi_events needs.
  ##   permno is the grouping key and must NOT be in .SDcols — data.table
  ##   handles it via by = permno and re-attaches it to the result.
  ##--------------------------------------------------------------------------##
  
  labelled <- prices_dt[,
                        fn_detect_csi_events(
                          .SD,
                          C        = C,
                          M        = M,
                          T        = T,
                          end_date = end_date
                        ),
                        by      = permno,
                        .SDcols = c("date", "ret_adj", "wealth_index", "running_peak", "drawdown")
  ]
  
  ##--------------------------------------------------------------------------##
  ## Stage 3: Collapse monthly status → annual (permno, year) panel
  ##--------------------------------------------------------------------------##
  
  annual_panel <- fn_build_annual_panel(labelled)
  
  ## Restrict label years to analysis window
  annual_panel <- annual_panel[year >= year(START_DATE) & year <= year(END_DATE)]
  
  return(annual_panel)
}

#==============================================================================#
# 4. Prepare data.table input
#==============================================================================#

dt_prices <- as.data.table(prices_monthly)[
  , .(permno, date, ret_adj)
][order(permno, date)]

## Censoring cutoff: zombie windows extending beyond END_DATE cannot be
## confirmed. We use END_DATE as the cutoff — fn_detect_csi_events handles
## the per-trigger comparison internally.
CENSOR_CUTOFF <- END_DATE

#==============================================================================#
# 5. Run base case
#==============================================================================#

cat("\n[05_CSI_Label.R] Running base case...\n")

labels_base <- fn_run_csi_pipeline(
  prices_dt = copy(dt_prices),
  C         = CSI_BASE$C,
  M         = CSI_BASE$M,
  T         = CSI_BASE$T,
  end_date  = CENSOR_CUTOFF,
  param_id  = "BASE"
)

labels_base[, param_id := "BASE"]

saveRDS(labels_base, PATH_LABELS_BASE)

prevalence_base <- mean(labels_base$y == 1, na.rm = TRUE)

cat(sprintf("[05_CSI_Label.R] Base case complete.\n"))
cat(sprintf("  Total (permno, year) observations : %d\n",  nrow(labels_base)))
cat(sprintf("  CSI events         (y = 1)        : %d\n",  sum(labels_base$y == 1,  na.rm = TRUE)))
cat(sprintf("  Clean observations (y = 0)        : %d\n",  sum(labels_base$y == 0,  na.rm = TRUE)))
cat(sprintf("  Zombie / censored  (y = NA)       : %d\n",  sum(is.na(labels_base$y))))
cat(sprintf("  CSI prevalence                    : %.2f%%\n", 100 * prevalence_base))

if (prevalence_base > MAX_IMPLOSION_RATE) {
  warning(sprintf(
    "[05_CSI_Label.R] WARNING: Base case prevalence %.2f%% exceeds MAX_IMPLOSION_RATE %.2f%%.",
    100 * prevalence_base, 100 * MAX_IMPLOSION_RATE
  ))
}

#==============================================================================#
# 6. Run full sensitivity grid (27 combinations)
#==============================================================================#

cat(sprintf("\n[05_CSI_Label.R] Running sensitivity grid (%d combinations)...\n",
            nrow(CSI_GRID)))

grid_results <- vector("list", nrow(CSI_GRID))
diagnostics  <- vector("list", nrow(CSI_GRID))

for (i in seq_len(nrow(CSI_GRID))) {
  
  params   <- CSI_GRID[i, ]
  param_id <- params$param_id
  
  result <- fn_run_csi_pipeline(
    prices_dt = copy(dt_prices),
    C         = params$C,
    M         = params$M,
    T         = params$T,
    end_date  = CENSOR_CUTOFF,
    param_id  = param_id
  )
  
  result[, param_id := param_id]
  
  ## Save individual grid result
  saveRDS(result, file.path(DIR_LABELS, paste0("labels_", param_id, ".rds")))
  
  prevalence <- mean(result$y == 1, na.rm = TRUE)
  
  diagnostics[[i]] <- data.table(
    param_id       = param_id,
    C              = params$C,
    M              = params$M,
    T              = params$T,
    n_obs          = nrow(result),
    n_csi          = sum(result$y == 1, na.rm = TRUE),
    n_clean        = sum(result$y == 0, na.rm = TRUE),
    n_na           = sum(is.na(result$y)),
    prevalence_pct = round(100 * prevalence, 3),
    above_threshold = prevalence > MAX_IMPLOSION_RATE
  )
  
  grid_results[[i]] <- result
  
  cat(sprintf("  [%d/%d] %s — prevalence: %.2f%% %s\n",
              i, nrow(CSI_GRID), param_id,
              100 * prevalence,
              if (prevalence > MAX_IMPLOSION_RATE) "WARNING: ABOVE THRESHOLD" else "ok"))
}

labels_all_grid <- rbindlist(grid_results)
csi_diagnostics <- rbindlist(diagnostics)

saveRDS(labels_all_grid, PATH_LABELS_GRID)
saveRDS(csi_diagnostics, PATH_LABELS_DIAG)

cat("\n[05_CSI_Label.R] Sensitivity grid complete.\n")

#==============================================================================#
# 7. Assertions on base case output
#==============================================================================#

cat("[05_CSI_Label.R] Running assertions...\n")

## A) All permno in labels are in universe
orphan_permno <- setdiff(labels_base$permno, universe$permno)
if (length(orphan_permno) > 0)
  stop(sprintf("[05_CSI_Label.R] ASSERTION FAILED: %d orphan permno in labels.",
               length(orphan_permno)))

## B) y values are only 0, 1, or NA
invalid_y <- labels_base[!is.na(y) & !y %in% c(0L, 1L)]
if (nrow(invalid_y) > 0)
  stop("[05_CSI_Label.R] ASSERTION FAILED: Invalid y values detected.")

## C) No duplicate (permno, year) rows
n_dup <- sum(duplicated(labels_base[, .(permno, year)]))
if (n_dup > 0)
  stop(sprintf("[05_CSI_Label.R] ASSERTION FAILED: %d duplicate (permno, year) rows.", n_dup))

## D) Year range within expected bounds
if (min(labels_base$year) < year(START_DATE) |
    max(labels_base$year) > year(END_DATE))
  stop("[05_CSI_Label.R] ASSERTION FAILED: Year range outside expected bounds.")

cat("[05_CSI_Label.R] All assertions passed.\n")

#==============================================================================#
# 8. Diagnostics — CSI events per year (replicates Figure 3 of paper)
#==============================================================================#

cat("[05_CSI_Label.R] Generating diagnostics plot...\n")

events_per_year <- labels_base[y == 1L, .N, by = year][order(year)]

p_events <- ggplot(events_per_year, aes(x = year, y = N)) +
  geom_col(fill = "#2c5f8a", width = 0.7) +
  labs(
    title    = "CSI Events per Year — Base Case",
    subtitle = sprintf("C = %.2f | M = %.2f | T = %d months",
                       CSI_BASE$C, CSI_BASE$M, CSI_BASE$T),
    x        = "Year",
    y        = "Number of CSI Events",
    caption  = "Replicates Figure 3 of Tewari, Galas & Treleaven (2024)"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  PATH_FIGURE_CSI,
  plot   = p_events,
  width  = PLOT_WIDTH,
  height = PLOT_HEIGHT,
  dpi    = PLOT_DPI
)

cat("\n[05_CSI_Label.R] Grid prevalence summary:\n")
print(csi_diagnostics[, .(param_id, C, M, T, prevalence_pct, above_threshold)])

#==============================================================================#
# 9. Final summary
#==============================================================================#

cat("\n[05_CSI_Label.R] ══════════════════════════════════════\n")
cat("  Base case label distribution:\n")
cat(sprintf("    y = 1  (CSI)      : %d\n",   sum(labels_base$y == 1,  na.rm = TRUE)))
cat(sprintf("    y = 0  (clean)    : %d\n",   sum(labels_base$y == 0,  na.rm = TRUE)))
cat(sprintf("    y = NA (excluded) : %d\n",   sum(is.na(labels_base$y))))
cat(sprintf("  Prevalence          : %.3f%%\n", 100 * prevalence_base))
cat(sprintf("  Grid combinations   : %d\n",   nrow(CSI_GRID)))
cat(sprintf("  Labels saved to     : %s\n",   DIR_LABELS))
cat("[05_CSI_Label.R] DONE:", format(Sys.time()), "\n")