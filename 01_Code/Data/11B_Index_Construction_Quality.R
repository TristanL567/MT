#==============================================================================#
#==== 11B_Quality.R ===========================================================#
#==== Quality Framework — Factor Overlays vs ML Risk Overlay =================#
#==============================================================================#
#
# PURPOSE:
#   Compare the ML exclusion overlay (from 11_Results.R) against three
#   classical factor-based screens applied as exclusion overlays.
#   Answers: does the ML model capture something simple linear screens miss,
#   or is the ML overlay a sophisticated proxy for low-vol / quality / momentum?
#
# FACTOR SCREENS (all implemented as exclusion overlays):
#   Low-Vol  : Exclude top 5% of universe by trailing 36m realised volatility
#               (highest-vol firms excluded — mirrors ML exclusion direction)
#   Quality  : Exclude top 5% by composite quality risk score
#               (worst quality excluded — composite of Altman Z2 percentile,
#               leverage percentile, ROA percentile — inverted so high = bad)
#   Momentum : Exclude top 5% by 12m price momentum LOSS
#               (worst momentum excluded — prior-year losers most at risk)
#
# COMPARISON LOGIC:
#   All screens applied to the same pseudo-Russell 3000 universe as ML overlay.
#   Same exclusion rate (5% base), same EW/CW weighting, same periods.
#   Returns and weights generated here; performance compared in 12_Index_Evaluation.
#
# APPENDIX — CONCENTRATED PORTFOLIOS:
#   Each screen also generates a 200-firm concentrated long portfolio
#   (safest / highest-quality firms selected). Appended to outputs.
#   Label: {factor}_long200_ew
#
# FACTOR CONSTRUCTION:
#   Low-vol  : trailing 36m monthly return standard deviation (annualised)
#              from PATH_PRICES_MONTHLY
#   Quality  : composite z-score of {altman_z2_pct, -leverage_pct, roa_pct}
#              from PATH_FEATURES_FUND. Higher composite = better quality.
#              Exclusion = worst (lowest composite) 5%.
#   Momentum : trailing 12m return (total, compounded) from monthly prices.
#              Exclusion = worst (lowest 12m return) 5%.
#
# OVERLAP ANALYSIS:
#   For each pair (ML model, factor screen), compute Jaccard similarity of
#   excluded firms per year. High overlap = ML screen is a factor proxy.
#   Low overlap = ML captures orthogonal information.
#
# INPUTS:
#   config.R
#   DIR_TABLES/index_weights.rds          ML overlay weights (from 11_Results.R)
#   DIR_TABLES/index_returns.rds          ML overlay returns
#   PATH_PRICES_MONTHLY
#   PATH_FEATURES_FUND
#
# OUTPUTS:
#   DIR_TABLES/quality_returns.rds        monthly returns (all factor strategies)
#   DIR_TABLES/quality_weights.rds        annual weights
#   DIR_TABLES/quality_performance.rds    performance table
#   DIR_TABLES/quality_overlap.rds        Jaccard overlap with ML strategies
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(arrow)
  library(scales)
  library(lubridate)
  library(tidyr)
})

cat("\n[11B_Quality.R] START:", format(Sys.time()), "\n")
FIGS <- fn_setup_figure_dirs()

## ── Parameters (match 11_Results.R) ─────────────────────────────────────────
UNIVERSE_SIZE  <- 3000L
MIN_MKTCAP_MM  <- 100
RF_ANNUAL      <- 0.03
UNIVERSE_MONTH <- 12L
REBAL_MONTHS   <- c(3L, 6L, 9L, 12L)
EXCL_RATE      <- 0.05        ## primary comparison rate
LONG_N         <- 200L        ## concentrated long portfolio size
INSAMPLE_START <- 1998L
OOS_END        <- 2024L
VOL_WINDOW     <- 36L         ## months for trailing volatility
MOM_WINDOW     <- 12L         ## months for momentum

#==============================================================================#
# 1. Load shared inputs
#==============================================================================#

cat("[11B] Loading inputs...\n")

monthly <- as.data.table(readRDS(PATH_PRICES_MONTHLY))
setnames(monthly, "ret_adj", "ret")
setnames(monthly, "mktcap",  "mkvalt")
monthly[, year  := year(date)]
monthly[, month := month(date)]
if (!inherits(monthly$date,"Date")) monthly[, date := as.Date(date)]
monthly[, ret := pmin(pmax(ret,-0.99,na.rm=TRUE),10,na.rm=TRUE)]

features <- as.data.table(readRDS(PATH_FEATURES_FUND))

## Rebuild annual universe (same as 11_Results.R)
dec_mv <- monthly[month==UNIVERSE_MONTH & !is.na(mkvalt),
                  .(mkvalt_dec=mkvalt[.N]), by=.(permno,year)]
universe_ann <- dec_mv[mkvalt_dec >= MIN_MKTCAP_MM]
universe_ann[, rank_mv := frank(-mkvalt_dec,ties.method="first"), by=year]
universe_ann <- universe_ann[rank_mv <= UNIVERSE_SIZE]

## Load ML weights for overlap analysis (M1 at 5% EW)
ml_weights <- readRDS(file.path(DIR_TABLES,"index_weights.rds"))
ml_returns <- readRDS(file.path(DIR_TABLES,"index_returns.rds"))

cat(sprintf("  Universe: %d firm-years | Monthly: %d rows\n",
            nrow(universe_ann),nrow(monthly)))

#==============================================================================#
# 2. Compute factor scores
#==============================================================================#

cat("\n[11B] Computing factor scores...\n")

## ── 2A. Low-volatility: trailing 36m annualised vol ──────────────────────────
cat("  [Low-Vol] Computing trailing 36m vol...\n")

setkey(monthly, permno, date)
monthly[, vol_36m := {
  r   <- ret
  n   <- length(r)
  vol <- rep(NA_real_, n)
  for (j in seq_len(n)) {
    if (j < VOL_WINDOW) next
    w_ret <- r[max(1, j-VOL_WINDOW+1):j]
    vol[j] <- sd(w_ret, na.rm=TRUE) * sqrt(12)
  }
  vol
}, by=permno]

## Take December vol as annual signal
lowvol_ann <- monthly[month==UNIVERSE_MONTH & !is.na(vol_36m),
                      .(permno, year, vol_36m)]

cat(sprintf("  Low-vol scores: %d firm-years\n", nrow(lowvol_ann)))

## ── 2B. Quality composite ─────────────────────────────────────────────────────
cat("  [Quality] Building composite quality risk score...\n")

## Quality risk: HIGH score = BAD firm (exclusion target — consistent with ML)
## Components:
##   1. altman_z2 percentile (INVERTED — low Z = high risk)
##   2. leverage percentile  (high leverage = high risk)
##   3. roa percentile       (INVERTED — low ROA = high risk)
## Composite = average of three standardised risk signals

qual_cols <- intersect(c("altman_z2","leverage","roa"), names(features))
cat(sprintf("  Quality features available: %s\n", paste(qual_cols,collapse=", ")))

if (length(qual_cols) >= 2L) {
  qual_dt <- features[, c("permno","year",qual_cols), with=FALSE]
  
  ## Rank within year → percentile (higher rank = worse quality = higher exclusion risk)
  if ("altman_z2" %in% qual_cols) {
    qual_dt[, altman_z2_risk := frank(-altman_z2,ties.method="average",na.last="keep")/.N,
            by=year]
  }
  if ("leverage" %in% qual_cols) {
    qual_dt[, leverage_risk := frank(leverage,ties.method="average",na.last="keep")/.N,
            by=year]
  }
  if ("roa" %in% qual_cols) {
    qual_dt[, roa_risk := frank(-roa,ties.method="average",na.last="keep")/.N,
            by=year]
  }
  
  risk_cols <- intersect(c("altman_z2_risk","leverage_risk","roa_risk"), names(qual_dt))
  qual_dt[, quality_risk := rowMeans(.SD, na.rm=TRUE), .SDcols=risk_cols]
  qual_ann <- qual_dt[!is.na(quality_risk), .(permno, year, quality_risk)]
} else {
  cat("  WARNING: Fewer than 2 quality features found — using altman_z2 only\n")
  qual_ann <- features[!is.na(altman_z2),
                       .(permno, year, quality_risk=frank(-altman_z2,ties.method="average",na.last="keep")/.N),
                       by=year][, .(permno,year,quality_risk)]
}
cat(sprintf("  Quality risk scores: %d firm-years\n", nrow(qual_ann)))

## ── 2C. Momentum: trailing 12m compounded return ─────────────────────────────
cat("  [Momentum] Computing trailing 12m returns...\n")

setkey(monthly, permno, date)
monthly[, mom_12m := {
  r   <- ret
  n   <- length(r)
  m12 <- rep(NA_real_, n)
  for (j in seq_len(n)) {
    if (j < MOM_WINDOW) next
    w_ret   <- r[max(1,j-MOM_WINDOW+1):(j-1)]  ## exclude current month (standard)
    m12[j]  <- prod(1+w_ret,na.rm=TRUE)-1
  }
  m12
}, by=permno]

mom_ann <- monthly[month==UNIVERSE_MONTH & !is.na(mom_12m),
                   .(permno, year, mom_12m)]
cat(sprintf("  Momentum scores: %d firm-years\n", nrow(mom_ann)))

#==============================================================================#
# 3. Build factor exclusion overlays
#==============================================================================#

cat("\n[11B] Building factor exclusion overlays...\n")

## For each factor, join to universe and apply exclusion
## EXCLUSION direction: high-risk (high vol, high quality_risk, low momentum) excluded

fn_factor_weights <- function(universe_ann, score_dt, score_col,
                              excl_high=TRUE,   ## TRUE = exclude highest scores
                              excl_rate=EXCL_RATE,
                              long_n=LONG_N,
                              factor_nm) {
  
  ## Join to universe: prediction at year t → portfolio in year t+1
  ann_f <- merge(universe_ann[,.(permno,year,mkvalt_dec)],
                 score_dt,
                 by=c("permno","year"), all.x=TRUE)
  
  ## Annual flags
  ann_f[, n_pred := sum(!is.na(get(score_col))), by=year]
  ann_f[!is.na(get(score_col)), rank_exc := {
    if (excl_high) frank(-get(score_col),ties.method="first")
    else            frank(get(score_col), ties.method="first")
  }, by=year]
  ## rank_exc = 1 → worst firm (to be excluded first)
  ann_f[, cutoff  := ceiling(n_pred * excl_rate)]
  ann_f[, flag    := !is.na(rank_exc) & rank_exc <= cutoff]
  ann_f[, port_year := year + 1L]
  
  ## Long portfolio (opposite end — safest firms)
  ann_f[!is.na(rank_exc), rank_long := {
    if (excl_high) frank(get(score_col), ties.method="first")  ## lowest = safest
    else            frank(-get(score_col), ties.method="first") ## highest = safest
  }, by=year]
  ann_f[, incl_long := !is.na(rank_long) & rank_long <= long_n]
  
  ## Quarter-end rebalancing (same cadence as 11_Results.R)
  q_dates_f <- monthly[month %in% REBAL_MONTHS,
                       .(qdate=max(date)), by=.(year,month)]
  setorder(q_dates_f,qdate)
  q_dates_f <- q_dates_f[year>=INSAMPLE_START & year<=OOS_END]
  
  wt_list <- list()
  for (i in seq_len(nrow(q_dates_f))) {
    q_yr  <- q_dates_f$year[i]
    q_mo  <- q_dates_f$month[i]
    qdate <- q_dates_f$qdate[i]
    
    ## Universe for this year × factor signals from year-1
    sig_yr <- ann_f[port_year == q_yr]
    ## Join to current December universe
    uni_q  <- universe_ann[year==q_yr, .(permno,mkvalt_dec)]
    sig_q  <- merge(uni_q, sig_yr[,.(permno,flag,incl_long)],
                    by="permno", all.x=TRUE)
    sig_q[is.na(flag), flag:=FALSE]
    sig_q[is.na(incl_long), incl_long:=FALSE]
    
    ## Exclusion overlay
    incl_ov <- sig_q[flag==FALSE]
    if (nrow(incl_ov)>0) {
      sm <- sum(incl_ov$mkvalt_dec,na.rm=TRUE)
      for (wt in c("ew","cw")) {
        wt_list[[paste0("ov_",wt,"_",i)]] <- data.table(
          permno=incl_ov$permno, mkvalt_dec=incl_ov$mkvalt_dec,
          qdate=qdate, q_year=q_yr, q_month=q_mo,
          factor=factor_nm, strategy="overlay", weighting=wt,
          w=if(wt=="ew") rep(1/nrow(incl_ov),nrow(incl_ov))
          else incl_ov$mkvalt_dec/sm)
      }
    }
    
    ## Long-only concentrated (appendix)
    incl_lo <- sig_q[incl_long==TRUE]
    if (nrow(incl_lo)>0) {
      sm2 <- sum(incl_lo$mkvalt_dec,na.rm=TRUE)
      for (wt in c("ew","cw")) {
        wt_list[[paste0("lo_",wt,"_",i)]] <- data.table(
          permno=incl_lo$permno, mkvalt_dec=incl_lo$mkvalt_dec,
          qdate=qdate, q_year=q_yr, q_month=q_mo,
          factor=factor_nm, strategy="long200", weighting=wt,
          w=if(wt=="ew") rep(1/nrow(incl_lo),nrow(incl_lo))
          else incl_lo$mkvalt_dec/sm2)
      }
    }
  }
  rbindlist(wt_list,use.names=TRUE)
}

## Build each factor
wt_lowvol  <- fn_factor_weights(universe_ann, lowvol_ann, "vol_36m",
                                excl_high=TRUE,  factor_nm="lowvol")
wt_quality <- fn_factor_weights(universe_ann, qual_ann,   "quality_risk",
                                excl_high=TRUE,  factor_nm="quality")
wt_momentum<- fn_factor_weights(universe_ann, mom_ann,    "mom_12m",
                                excl_high=FALSE, factor_nm="momentum")

quality_weights <- rbindlist(list(wt_lowvol, wt_quality, wt_momentum),
                             use.names=TRUE)
saveRDS(quality_weights, file.path(DIR_TABLES,"quality_weights.rds"))
cat(sprintf("  Factor weights: %d rows\n", nrow(quality_weights)))

#==============================================================================#
# 4. Compute factor monthly returns
#==============================================================================#

cat("\n[11B] Computing factor monthly returns...\n")

fac_strats <- unique(quality_weights[,.(factor,strategy,weighting)])
fac_ret_list <- vector("list",nrow(fac_strats))

for (i in seq_len(nrow(fac_strats))) {
  fs <- fac_strats[i]
  w_f <- quality_weights[factor==fs$factor &
                           strategy==fs$strategy &
                           weighting==fs$weighting,
                         .(permno,qdate,w)]
  rdates <- sort(unique(w_f$qdate))
  rel_p  <- unique(w_f$permno)
  m_sub  <- monthly[permno %in% rel_p, .(permno,date,year,month,ret)]
  m_sub[, aqd := rdates[findInterval(date,rdates,left.open=FALSE)]]
  m_sub  <- m_sub[!is.na(aqd)]
  setnames(w_f,"qdate","aqd")
  m_f    <- merge(m_sub, w_f, by=c("permno","aqd"), all.x=FALSE)
  m_f    <- m_f[!is.na(ret)&!is.na(w)]
  
  fac_ret_list[[i]] <- m_f[,.(
    port_ret   = sum(w*ret,na.rm=TRUE),
    n_holdings = uniqueN(permno),
    factor     = fs$factor,
    strategy   = fs$strategy,
    weighting  = fs$weighting
  ), by=.(date,year,month)]
}

## Add benchmark (from ML returns)
bench_ret <- ml_returns[model_key=="bench" & weighting=="ew",
                        .(date,year,month,port_ret,n_holdings,
                          factor="bench",strategy="bench",weighting="ew")]

quality_returns <- rbindlist(c(fac_ret_list, list(bench_ret)),
                             use.names=TRUE, fill=TRUE)
setorder(quality_returns, factor, strategy, weighting, date)
saveRDS(quality_returns, file.path(DIR_TABLES,"quality_returns.rds"))
cat(sprintf("  Factor returns: %d rows\n",nrow(quality_returns)))

#==============================================================================#
# 5. Factor performance table
#==============================================================================#

cat("\n[11B] Performance metrics...\n")

fn_perf <- function(rv,rf=RF_ANNUAL) {
  rv <- rv[is.finite(rv)]; if(length(rv)<12) return(NULL)
  ny <- length(rv)/12; rfm <- (1+rf)^(1/12)-1
  cum <- prod(1+rv)-1; cagr <- (1+cum)^(1/ny)-1
  vol <- sd(rv)*sqrt(12); exc <- rv-rfm
  sh  <- mean(exc)/sd(exc)*sqrt(12)
  ddr <- exc[rv<rfm]
  srt <- if(length(ddr)>1) mean(exc)/(sd(ddr)*sqrt(12)) else NA_real_
  ci  <- cumprod(1+rv); pk <- cummax(ci); mdd <- min((ci-pk)/pk)
  cal <- if(mdd<0) cagr/abs(mdd) else NA_real_
  data.frame(n_months=length(rv),cum_ret=round(cum,4),cagr=round(cagr,4),
             vol=round(vol,4),sharpe=round(sh,4),sortino=round(srt,4),
             max_dd=round(mdd,4),calmar=round(cal,4))
}

PERIODS_Q <- list(
  insample=c(INSAMPLE_START,year(TRAIN_END)),
  test    =c(year(TEST_START),year(TEST_END)),
  oos     =c(year(OOS_START),OOS_END),
  full    =c(INSAMPLE_START,OOS_END)
)

qperf_rows <- list()
for (i in seq_len(nrow(fac_strats))) {
  fs  <- fac_strats[i]
  rdt <- quality_returns[factor==fs$factor &
                           strategy==fs$strategy &
                           weighting==fs$weighting]
  for (pnm in names(PERIODS_Q)) {
    yr  <- PERIODS_Q[[pnm]]
    sub <- rdt[year>=yr[1]&year<=yr[2]]
    pf  <- fn_perf(sub$port_ret); if(is.null(pf)) next
    pf$factor <- fs$factor; pf$strategy <- fs$strategy
    pf$weighting <- fs$weighting; pf$period <- pnm
    qperf_rows[[length(qperf_rows)+1]] <- pf
  }
}

quality_perf <- rbindlist(qperf_rows,fill=TRUE)
saveRDS(quality_perf, file.path(DIR_TABLES,"quality_performance.rds"))

## Console summary — OOS, EW, overlay
cat("\n  Factor overlay performance (OOS, EW):\n")
qoos <- quality_perf[period=="oos"&strategy=="overlay"&weighting=="ew"]
boos <- fn_perf(quality_returns[factor=="bench"&strategy=="bench"&
                                  weighting=="ew"&year>=year(OOS_START)&
                                  year<=OOS_END,port_ret])
if(!is.null(boos)) cat(sprintf("  Benchmark : CAGR=%+.2f%% | Sharpe=%.3f | MaxDD=%.2f%%\n",
                               boos$cagr*100,boos$sharpe,boos$max_dd*100))
for(j in seq_len(nrow(qoos))) {
  r <- qoos[j]
  cat(sprintf("  %-10s: CAGR=%+.2f%% | Sharpe=%.3f | MaxDD=%.2f%%\n",
              r$factor,r$cagr*100,r$sharpe,r$max_dd*100))
}

#==============================================================================#
# 6. Overlap analysis — ML exclusions vs factor exclusions
#==============================================================================#

cat("\n[11B] Computing exclusion overlap (Jaccard)...\n")

## Get excluded firms per year for each factor (December rebalance)
fn_excl_set <- function(wt_dt, fac, strat, wt_type) {
  uni_yr <- universe_ann[,.(permno,year)]
  incl   <- wt_dt[factor==fac & strategy==strat & weighting==wt_type &
                    q_month==UNIVERSE_MONTH, .(permno,q_year)]
  setnames(incl,"q_year","year")
  merged <- merge(uni_yr, incl, by=c("permno","year"), all.x=TRUE)
  merged[is.na(permno)]  ## never reached — keep anti-join approach
  ## anti-join: universe minus included = excluded
  excl_dt <- uni_yr[!incl, on=c("permno","year")]
  excl_dt
}

fn_jaccard <- function(a,b) {
  n_i <- nrow(merge(a,b,by=c("permno","year")))
  n_u <- nrow(unique(rbindlist(list(a,b))))
  if(n_u==0) return(NA_real_)
  n_i/n_u
}

## ML exclusions (M1, 5%, EW)
ml_excl_sets <- lapply(
  intersect(c("fund","raw","bucket","structural"), names(PREDS)),
  function(key) {
    incl <- ml_weights[model_key==key & excl_rate=="5pct" & weighting=="ew" &
                         q_month==UNIVERSE_MONTH, .(permno,q_year)]
    setnames(incl,"q_year","year")
    excl <- universe_ann[,.(permno,year)][!incl, on=c("permno","year")]
    list(key=key, excl=excl)
  }
)

## Factor exclusion sets
factor_excl <- lapply(c("lowvol","quality","momentum"), function(fac) {
  incl <- quality_weights[factor==fac & strategy=="overlay" & weighting=="ew" &
                            q_month==UNIVERSE_MONTH, .(permno,q_year)]
  setnames(incl,"q_year","year")
  excl <- universe_ann[,.(permno,year)][!incl, on=c("permno","year")]
  list(factor=fac, excl=excl)
})

## Pairwise Jaccard
overlap_rows <- list()
for (ml in ml_excl_sets) {
  for (fac in factor_excl) {
    jacc <- fn_jaccard(ml$excl, fac$excl)
    overlap_rows[[length(overlap_rows)+1]] <- data.frame(
      ml_model   = ml$key,
      factor     = fac$factor,
      jaccard    = round(jacc,4),
      n_ml_excl  = nrow(ml$excl),
      n_fac_excl = nrow(fac$excl),
      stringsAsFactors=FALSE
    )
  }
}
quality_overlap <- rbindlist(overlap_rows,fill=TRUE)
saveRDS(quality_overlap, file.path(DIR_TABLES,"quality_overlap.rds"))

cat("\n  Exclusion overlap (Jaccard) — ML vs factor screens:\n")
print(dcast(quality_overlap, ml_model~factor, value.var="jaccard"),
      row.names=FALSE)
cat(sprintf("\n  Interpretation: Jaccard=0 = no overlap | Jaccard=1 = identical exclusion sets\n"))
cat(sprintf("  Low Jaccard suggests ML captures orthogonal information vs the factor screen\n"))

#==============================================================================#
# 7. Key plots
#==============================================================================#

cat("\n[11B] Plots...\n")

FACTOR_COLS <- c(bench="#9E9E9E",lowvol="#FF9800",quality="#1B5E20",momentum="#9C27B0")
FACTOR_LABS <- c(bench="Benchmark",lowvol="Low-Vol Excl.",
                 quality="Quality Excl.",momentum="Momentum Excl.")

## ── 7A. Factor overlays vs benchmark — OOS cumulative (EW) ──────────────────

oos_ret_q <- quality_returns[year>=year(OOS_START) & year<=OOS_END &
                               weighting=="ew" &
                               ((strategy=="overlay") | (factor=="bench"))]
setorder(oos_ret_q, factor,strategy,date)
oos_ret_q[, cum_idx := cumprod(1+port_ret), by=.(factor,strategy)]

p_factor_oos <- ggplot(oos_ret_q,
                       aes(x=date,y=cum_idx,colour=factor,group=factor)) +
  geom_line(linewidth=0.9) +
  scale_colour_manual(values=FACTOR_COLS, labels=FACTOR_LABS) +
  scale_y_continuous(labels=dollar_format(prefix="$")) +
  scale_x_date(date_breaks="6 months",date_labels="%Y-%m") +
  labs(title="Factor Exclusion Overlays vs Benchmark — OOS 2020-2024 (EW, 5%)",
       subtitle="All screens exclude the worst 5% of universe by factor score",
       x=NULL, y="Portfolio Value ($1)", colour=NULL) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1))

ggsave(file.path(FIGS$index_general,"factor_overlays_oos.png"), p_factor_oos,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  factor_overlays_oos.png saved.\n")

## ── 7B. ML (M1) vs factor overlays — full period EW ─────────────────────────

m1_full <- ml_returns[model_key=="fund"&excl_rate=="5pct"&weighting=="ew",
                      .(date,year,month,port_ret,n_holdings,factor="ml_m1",strategy="overlay",weighting="ew")]

comp_dt <- rbindlist(list(
  quality_returns[strategy %in% c("overlay","bench") & weighting=="ew"],
  m1_full), use.names=TRUE, fill=TRUE)
setorder(comp_dt,factor,date)
comp_dt[, cum_idx := cumprod(1+port_ret), by=factor]

COMP_COLS <- c(FACTOR_COLS, ml_m1="#1565C0")
COMP_LABS <- c(FACTOR_LABS, ml_m1="M1 ML Overlay")

p_ml_vs_factor <- ggplot(comp_dt[factor %in% names(COMP_COLS)],
                         aes(x=date,y=cum_idx,colour=factor,group=factor)) +
  geom_line(linewidth=0.85) +
  geom_vline(xintercept=as.numeric(as.Date("2016-01-01")),
             linetype="dashed",colour="grey40") +
  scale_colour_manual(values=COMP_COLS, labels=COMP_LABS) +
  scale_y_continuous(labels=dollar_format(prefix="$")) +
  scale_x_date(date_breaks="2 years",date_labels="%Y") +
  labs(title="ML Overlay vs Factor Screens — Full Period (EW, 5%)",
       subtitle="Dashed = test period start (2016)",
       x=NULL, y="Portfolio Value ($1)", colour=NULL) +
  theme_minimal(base_size=12) +
  theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1))

ggsave(file.path(FIGS$index_general,"ml_vs_factor_full.png"), p_ml_vs_factor,
       width=PLOT_WIDTH*1.2, height=PLOT_HEIGHT, dpi=PLOT_DPI)
cat("  ml_vs_factor_full.png saved.\n")

## ── 7C. Jaccard overlap heatmap ──────────────────────────────────────────────

if (nrow(quality_overlap)>0) {
  quality_overlap[, ml_label := MODEL_LABELS[ml_model]]
  p_jacc <- ggplot(quality_overlap,
                   aes(x=factor,y=ml_label,fill=jaccard)) +
    geom_tile(colour="white",linewidth=0.5) +
    geom_text(aes(label=round(jaccard,3)),size=3.5,colour="white") +
    scale_fill_viridis(option="mako",limits=c(0,1),name="Jaccard") +
    labs(title="Exclusion Set Overlap: ML Models vs Factor Screens",
         subtitle="Jaccard similarity of excluded firms (December, 5% excl rate) | 0=no overlap, 1=identical",
         x="Factor Screen", y="ML Model") +
    theme_minimal(base_size=12)
  
  ggsave(file.path(FIGS$index_general,"jaccard_overlap_heatmap.png"), p_jacc,
         width=PLOT_WIDTH, height=PLOT_HEIGHT*0.8, dpi=PLOT_DPI)
  cat("  jaccard_overlap_heatmap.png saved.\n")
}

cat(sprintf("\n[11B_Quality.R] DONE: %s\n", format(Sys.time())))