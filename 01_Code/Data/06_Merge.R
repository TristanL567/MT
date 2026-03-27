## ── PASTE THIS OVER SECTION 3 IN 06_Merge.R ─────────────────────────────────
##
## Changes from original:
##   - gdp_growth, cpi_inflation, indpro_growth are now already decimal
##     YoY ratios in macro_monthly (computed in 04_Macro.R). The annual
##     mean of a monthly growth rate = average monthly rate during year t.
##     Label in thesis: "mean monthly YoY growth rate during year t".
##   - d_unrate, d_hy_spread, d_vix are new YoY change columns.
##     Annual mean aggregation is correct — captures the average direction
##     of change across the year.
##   - recession: max within year (unchanged — 1 if any month was recession).
##   - All level variables (unrate, fedfunds, gs10, hy_spread, vix,
##     term_spread, cpi, indpro): annual mean (unchanged).
##   - gdp level: annual mean (unchanged — used for log_at style size proxy
##     in macro interaction; growth rate now separate).

cat("[06_Merge.R] Collapsing monthly macro to annual...\n")

macro_annual <- macro_monthly |>
  mutate(year = year(date)) |>
  group_by(year) |>
  summarise(
    
    ##── Level variables — annual mean ──────────────────────────────────────
    gdp           = mean(gdp,           na.rm = TRUE),
    unrate        = mean(unrate,        na.rm = TRUE),
    fedfunds      = mean(fedfunds,      na.rm = TRUE),
    gs10          = mean(gs10,          na.rm = TRUE),
    term_spread   = mean(term_spread,   na.rm = TRUE),
    hy_spread     = mean(hy_spread,     na.rm = TRUE),
    vix           = mean(vix,           na.rm = TRUE),
    cpi           = mean(cpi,           na.rm = TRUE),
    indpro        = mean(indpro,        na.rm = TRUE),
    
    ##── YoY ratio changes — annual mean of monthly values ──────────────────
    ## Interpretation: average YoY growth rate prevailing during year t.
    ## Units: decimal (0.025 = 2.5% growth). Computed in 04_Macro.R.
    gdp_growth    = mean(gdp_growth,    na.rm = TRUE),
    cpi_inflation = mean(cpi_inflation, na.rm = TRUE),
    indpro_growth = mean(indpro_growth, na.rm = TRUE),
    d_vix         = mean(d_vix,         na.rm = TRUE),
    
    ##── YoY pp changes — annual mean of monthly values ─────────────────────
    ## Interpretation: average direction of change in unemployment/credit
    ## spreads during year t. Positive = conditions deteriorating.
    d_unrate      = mean(d_unrate,      na.rm = TRUE),
    d_hy_spread   = mean(d_hy_spread,   na.rm = TRUE),
    
    ##── Recession — 1 if any month in the year was a recession ─────────────
    recession     = as.integer(max(recession, na.rm = TRUE)),
    
    .groups = "drop"
  )

macro_annual <- macro_annual |>
  mutate(across(where(is.numeric), ~if_else(is.nan(.x), NA_real_, .x)))

cat(sprintf("  Annual macro obs: %d rows, years %d–%d\n",
            nrow(macro_annual),
            min(macro_annual$year),
            max(macro_annual$year)))

## Update macro_cols assertion vector downstream (section 8E):
## Add d_unrate, d_hy_spread, d_vix to the check list
macro_cols <- c("gdp_growth", "unrate", "fedfunds", "term_spread",
                "hy_spread", "vix", "cpi_inflation", "indpro_growth",
                "recession", "d_unrate", "d_hy_spread", "d_vix")
