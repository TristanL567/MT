#==============================================================================#
#==== 11_Results.R ============================================================#
#==== Crash-Filtered Index Construction — All 12 Models =======================#
#==============================================================================#
#
# PURPOSE:
#   Construct and backtest crash-filtered equity indices for all 12 AutoGluon
#   models (M1-M4, B1-B4, S1-S4). Predicted probability scores rank-exclude
#   firms from a pseudo-Russell 3000 universe at four FPR-equivalent rates.
#   Outputs raw returns and weights consumed by 12_Index_Evaluation.R.
#
# INDEX DESIGN:
#   Universe   : Top 3000 by December market cap, refreshed annually.
#   Rebalancing: Quarterly (Mar/Jun/Sep/Dec).
#                Signal frequency: ANNUAL (Compustat-based predictions).
#                Universe membership: updates at December only.
#                Design note: quarterly cadence captures delistings / new
#                entrants; exclusion flags are stable within each year since
#                the underlying annual predictions do not change intra-year.
#                This is intentional and matches the data generation frequency.
#   Weighting  : Equal-weight (EW) and market-cap-weight (CW).
#   Exclusion  : Rank-based — top X% by predicted probability excluded.
#                Four rates: 1%, 3%, 5%, 10% (FPR-equivalent thresholds).
#                Firms with no prediction are never excluded (conservative).
#   Labels     : Not required. Probabilities only — valid across all periods
#                including OOS 2020-2024 where bucket/structural labels are
#                right-censored.
#
# STRATEGY MATRIX:
#   1 benchmark + 12 models x 4 rates x 2 weightings = 97 strategies
#
# PERIODS:
#   In-sample : 1998-2015   (CV fold predictions — less strict, documented)
#   Test      : 2016-2019   (honest evaluation period)
#   OOS       : 2020-2024   (live — no labels exist)
#
# INPUTS:
#   config.R
#   PATH_PRICES_MONTHLY
#   DIR_TABLES/ag_{MODEL}/ag_preds_test.parquet
#   DIR_TABLES/ag_{MODEL}/ag_preds_oos.parquet
#   DIR_TABLES/ag_{MODEL}/ag_cv_results.parquet     (optional, in-sample)
#
# OUTPUTS:
#   DIR_TABLES/index_returns.rds
#   DIR_TABLES/index_weights.rds
#   DIR_TABLES/index_performance.rds
#   DIR_TABLES/index_exclusion_summary.rds
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(arrow)
  library(scales)
  library(lubridate)
})

cat("\n[11_Results.R] START:", format(Sys.time()), "\n")
FIGS <- fn_setup_figure_dirs()

## ── Parameters ───────────────────────────────────────────────────────────────
UNIVERSE_SIZE  <- 3000L
MIN_MKTCAP_MM  <- 100
RF_ANNUAL      <- 0.03
REBAL_MONTHS   <- c(3L, 6L, 9L, 12L)
UNIVERSE_MONTH <- 12L
TC_BPS         <- 0L

EXCL_RATES <- c("1pct"=0.01, "3pct"=0.03, "5pct"=0.05, "10pct"=0.10)

INSAMPLE_START <- 1998L
OOS_END        <- 2024L

#==============================================================================#
# 1. Monthly prices
#==============================================================================#

cat("[11] Loading monthly prices...\n")
monthly <- as.data.table(readRDS(PATH_PRICES_MONTHLY))
setnames(monthly, "ret_adj", "ret")
setnames(monthly, "mktcap",  "mkvalt")
monthly[, year  := year(date)]
monthly[, month := month(date)]
if (!inherits(monthly$date,"Date")) monthly[, date := as.Date(date)]
monthly[, ret := pmin(pmax(ret,-0.99,na.rm=TRUE),10,na.rm=TRUE)]
cat(sprintf("  %d rows | %d permnos | %d-%d\n",
            nrow(monthly),uniqueN(monthly$permno),
            min(monthly$year),max(monthly$year)))

#==============================================================================#
# 2. Annual universe
#==============================================================================#

cat("[11] Building annual universe...\n")
dec_mv <- monthly[month==UNIVERSE_MONTH & !is.na(mkvalt),
                  .(mkvalt_dec=mkvalt[.N]), by=.(permno,year)]
universe_ann <- dec_mv[mkvalt_dec >= MIN_MKTCAP_MM]
universe_ann[, rank_mv := frank(-mkvalt_dec,ties.method="first"), by=year]
universe_ann <- universe_ann[rank_mv <= UNIVERSE_SIZE]
cat(sprintf("  %d firm-years | avg %.0f/yr | %d-%d\n",
            nrow(universe_ann),
            nrow(universe_ann)/uniqueN(universe_ann$year),
            min(universe_ann$year),max(universe_ann$year)))

#==============================================================================#
# 3. Load all 12 model predictions
#==============================================================================#

cat("[11] Loading predictions...\n")
SRC_PRI <- c(oos=1L,test=2L,boundary=3L,cv=4L)
PREDS   <- list()

for (key in MODEL_KEYS_ALL) {
  tdir  <- file.path(DIR_TABLES, paste0("ag_",key))
  fmap  <- c(cv      =file.path(tdir,"ag_cv_results.parquet"),
             boundary=file.path(tdir,"ag_preds_train_boundary.parquet"),
             test    =file.path(tdir,"ag_preds_test.parquet"),
             oos     =file.path(tdir,"ag_preds_oos.parquet"))
  
  parts <- Filter(Negate(is.null), lapply(names(fmap), function(nm) {
    if (!file.exists(fmap[[nm]])) return(NULL)
    dt <- as.data.table(arrow::read_parquet(fmap[[nm]]))
    dt[, src:=nm][, .(permno,year,p_csi,src)]
  }))
  if (length(parts)==0) { cat(sprintf("  [%s] skip\n",key)); next }
  
  comb <- rbindlist(parts)
  comb[, src_rank := SRC_PRI[src]]
  setorder(comb,permno,year,src_rank)
  comb <- comb[!duplicated(comb[,.(permno,year)])]
  PREDS[[key]] <- comb[,.(permno,year,p_csi)]
  cat(sprintf("  [%-30s] %d rows | %d-%d\n",
              MODEL_LABELS[[key]],nrow(PREDS[[key]]),
              min(PREDS[[key]]$year),max(PREDS[[key]]$year)))
}
cat(sprintf("  Loaded: %d/%d models\n",length(PREDS),length(MODEL_KEYS_ALL)))

preds_stack <- rbindlist(lapply(names(PREDS), function(k)
  PREDS[[k]][,model_key:=k]), use.names=TRUE)

#==============================================================================#
# 4. Build quarterly weights
#==============================================================================#

cat("\n[11] Building quarterly weights...\n")

q_dates <- monthly[month %in% REBAL_MONTHS,
                   .(qdate=max(date)), by=.(year,month)]
setorder(q_dates,qdate)
q_dates <- q_dates[year>=INSAMPLE_START & year<=OOS_END]
N_Q     <- nrow(q_dates)

w_list <- list()

for (i in seq_len(N_Q)) {
  q_yr  <- q_dates$year[i]
  q_mo  <- q_dates$month[i]
  qdate <- q_dates$qdate[i]
  
  uni_q <- universe_ann[year==q_yr, .(permno,mkvalt_dec)]
  if (nrow(uni_q)==0) next
  n_u   <- nrow(uni_q)
  
  ## Benchmark
  for (wt in c("ew","cw")) {
    w_list[[paste0("bench_",wt,"_",i)]] <- data.table(
      permno=uni_q$permno, mkvalt_dec=uni_q$mkvalt_dec,
      qdate=qdate, q_year=q_yr, q_month=q_mo,
      model_key="bench", excl_rate="none", weighting=wt,
      w=if(wt=="ew") rep(1/n_u,n_u)
      else uni_q$mkvalt_dec/sum(uni_q$mkvalt_dec))
  }
  
  ## Overlay strategies
  preds_yr <- preds_stack[year==q_yr-1L]
  
  for (key in names(PREDS)) {
    p_k <- preds_yr[model_key==key, .(permno,p_csi)]
    if (nrow(p_k)==0) next
    u_p <- merge(uni_q, p_k, by="permno", all.x=TRUE)
    n_p <- sum(!is.na(u_p$p_csi))
    if (n_p==0) next
    u_p[, rd := frank(-p_csi,ties.method="first",na.last="keep")]
    
    for (rn in names(EXCL_RATES)) {
      co   <- ceiling(n_p * EXCL_RATES[[rn]])
      incl <- u_p[is.na(rd)|rd>co]
      if (nrow(incl)==0) next
      sm   <- sum(incl$mkvalt_dec,na.rm=TRUE)
      for (wt in c("ew","cw")) {
        w_list[[paste0(key,"_",rn,"_",wt,"_",i)]] <- data.table(
          permno=incl$permno, mkvalt_dec=incl$mkvalt_dec,
          qdate=qdate, q_year=q_yr, q_month=q_mo,
          model_key=key, excl_rate=rn, weighting=wt,
          w=if(wt=="ew") rep(1/nrow(incl),nrow(incl))
          else incl$mkvalt_dec/sm)
      }
    }
  }
  if (i%%20==0||i==N_Q) cat(sprintf("  %d/%d\n",i,N_Q))
}

weights_all <- rbindlist(w_list, use.names=TRUE)
setorder(weights_all, model_key,excl_rate,weighting,qdate,permno)
saveRDS(weights_all, file.path(DIR_TABLES,"index_weights.rds"))
cat(sprintf("  Weights: %d rows saved\n",nrow(weights_all)))

#==============================================================================#
# 5. Monthly portfolio returns
#==============================================================================#

cat("\n[11] Computing monthly returns...\n")
strats <- unique(weights_all[,.(model_key,excl_rate,weighting)])
N_S    <- nrow(strats)
ret_list <- vector("list",N_S)

for (i in seq_len(N_S)) {
  sk  <- strats[i]
  w_s <- weights_all[model_key==sk$model_key &
                       excl_rate==sk$excl_rate &
                       weighting==sk$weighting,
                     .(permno,qdate,w)]
  rdates <- sort(unique(w_s$qdate))
  rel_p  <- unique(w_s$permno)
  m_sub  <- monthly[permno %in% rel_p, .(permno,date,year,month,ret)]
  m_sub[, aqd := rdates[findInterval(date,rdates,left.open=FALSE)]]
  m_sub  <- m_sub[!is.na(aqd)]
  ## rename for merge
  setnames(w_s,"qdate","aqd")
  m_s    <- merge(m_sub, w_s, by=c("permno","aqd"), all.x=FALSE)
  m_s    <- m_s[!is.na(ret)&!is.na(w)]
  if (TC_BPS>0) m_s[month %in% REBAL_MONTHS, ret:=ret-TC_BPS/10000]
  
  ret_list[[i]] <- m_s[,.(
    port_ret   = sum(w*ret,na.rm=TRUE),
    n_holdings = uniqueN(permno),
    model_key  = sk$model_key,
    excl_rate  = sk$excl_rate,
    weighting  = sk$weighting
  ), by=.(date,year,month)]
  
  if (i%%50==0||i==N_S) cat(sprintf("  %d/%d strategies\n",i,N_S))
}

port_returns <- rbindlist(ret_list)
setorder(port_returns,model_key,excl_rate,weighting,date)
saveRDS(port_returns, file.path(DIR_TABLES,"index_returns.rds"))
cat(sprintf("  Returns: %d rows\n",nrow(port_returns)))

#==============================================================================#
# 6. Performance table
#==============================================================================#

cat("\n[11] Performance metrics...\n")

fn_perf <- function(rv, rf=RF_ANNUAL) {
  rv <- rv[is.finite(rv)]
  if (length(rv)<12) return(NULL)
  ny  <- length(rv)/12; rfm <- (1+rf)^(1/12)-1
  cum <- prod(1+rv)-1;  cagr <- (1+cum)^(1/ny)-1
  vol <- sd(rv)*sqrt(12); exc <- rv-rfm
  sh  <- mean(exc)/sd(exc)*sqrt(12)
  ddr <- exc[rv<rfm]
  srt <- if(length(ddr)>1) mean(exc)/(sd(ddr)*sqrt(12)) else NA_real_
  ci  <- cumprod(1+rv); pk <- cummax(ci)
  mdd <- min((ci-pk)/pk)
  cal <- if(mdd<0) cagr/abs(mdd) else NA_real_
  data.frame(n_months=length(rv),cum_ret=round(cum,4),cagr=round(cagr,4),
             vol=round(vol,4),sharpe=round(sh,4),sortino=round(srt,4),
             max_dd=round(mdd,4),calmar=round(cal,4),
             win_rate=round(mean(rv>0),4))
}

PERIODS_P <- list(
  insample=c(INSAMPLE_START,year(TRAIN_END)),
  test    =c(year(TEST_START),year(TEST_END)),
  oos     =c(year(OOS_START),OOS_END),
  full    =c(INSAMPLE_START,OOS_END)
)

perf_rows <- list()
for (i in seq_len(N_S)) {
  sk  <- strats[i]
  rdt <- port_returns[model_key==sk$model_key &
                        excl_rate==sk$excl_rate &
                        weighting==sk$weighting]
  for (pnm in names(PERIODS_P)) {
    yr  <- PERIODS_P[[pnm]]
    sub <- rdt[year>=yr[1]&year<=yr[2]]
    pf  <- fn_perf(sub$port_ret); if(is.null(pf)) next
    pf$model_key <- sk$model_key; pf$excl_rate <- sk$excl_rate
    pf$weighting <- sk$weighting; pf$period     <- pnm
    pf$track     <- MODEL_TRACK[[sk$model_key]] %||% "—"
    pf$short     <- MODEL_SHORTS[[sk$model_key]] %||% sk$model_key
    perf_rows[[length(perf_rows)+1]] <- pf
  }
}
`%||%` <- function(a,b) if(is.null(a)||length(a)==0) b else a
perf_all <- rbindlist(perf_rows, fill=TRUE)
saveRDS(perf_all, file.path(DIR_TABLES,"index_performance.rds"))
cat("  index_performance.rds saved.\n")

## Console summary
cat("\n  OOS (EW, 5%) summary:\n")
oos5 <- perf_all[period=="oos"&excl_rate %in% c("none","5pct")&weighting=="ew"]
setorder(oos5,track,short)
br   <- oos5[model_key=="bench"]
if(nrow(br)>0) cat(sprintf("  BENCH   : CAGR=%+.2f%% | Sharpe=%.3f | MaxDD=%.2f%%\n",
                           br$cagr*100,br$sharpe,br$max_dd*100))
for(j in seq_len(nrow(oos5[model_key!="bench"]))) {
  r <- oos5[model_key!="bench"][j]
  cat(sprintf("  %-8s: CAGR=%+.2f%% | Sharpe=%.3f | MaxDD=%.2f%%\n",
              r$short,r$cagr*100,r$sharpe,r$max_dd*100))
}

#==============================================================================#
# 7. Exclusion diagnostics
#==============================================================================#

excl_d <- weights_all[q_month==UNIVERSE_MONTH,
                      .(n_included=.N), by=.(model_key,excl_rate,weighting,q_year)]
uni_sz <- universe_ann[,.(n_universe=.N),by=year]
excl_d <- merge(excl_d,uni_sz,by.x="q_year",by.y="year",all.x=TRUE)
excl_d[, n_excluded:=n_universe-n_included]
excl_d[, excl_pct  :=round(n_excluded/n_universe*100,2)]
saveRDS(excl_d, file.path(DIR_TABLES,"index_exclusion_summary.rds"))
cat("  index_exclusion_summary.rds saved.\n")

#==============================================================================#
# 8. Core plots
#==============================================================================#

cat("\n[11] Core plots...\n")

## Benchmark cumulative
bew <- port_returns[model_key=="bench"&weighting=="ew"][order(date)]
bew[, cum_idx := cumprod(1+port_ret)]
p_b <- ggplot(bew,aes(x=date,y=cum_idx)) +
  geom_line(colour="#9E9E9E",linewidth=1) +
  geom_vline(xintercept=as.numeric(as.Date("2016-01-01")),
             linetype="dashed",colour="#1565C0") +
  geom_vline(xintercept=as.numeric(as.Date("2020-01-01")),
             linetype="dotted",colour="#E53935") +
  scale_y_continuous(labels=dollar_format(prefix="$")) +
  scale_x_date(date_breaks="2 years",date_labels="%Y") +
  labs(title=sprintf("Benchmark — Pseudo-Russell %d (EW)",UNIVERSE_SIZE),
       subtitle="Annual universe | Quarterly rebalancing | Annual signal",
       x=NULL,y="Portfolio Value ($1)") +
  theme_minimal(base_size=12)
ggsave(file.path(FIGS$index_general,"benchmark_cumulative.png"),p_b,
       width=PLOT_WIDTH*1.2,height=PLOT_HEIGHT,dpi=PLOT_DPI)

## M1 exclusion rate comparison
RATE_COLS <- c(none="#9E9E9E","1pct"="#BBDEFB","3pct"="#1E88E5",
               "5pct"="#0D47A1","10pct"="#E53935")
RATE_LABS <- c(none="Benchmark","1pct"="Excl 1%","3pct"="Excl 3%",
               "5pct"="Excl 5%","10pct"="Excl 10%")

if ("fund" %in% names(PREDS)) {
  m1r <- rbindlist(list(
    bew[,.(date,year,port_ret,model_key="bench",excl_rate="none")],
    port_returns[model_key=="fund"&weighting=="ew",
                 .(date,year,port_ret,model_key,excl_rate)]))
  setorder(m1r,model_key,excl_rate,date)
  m1r[, cum_idx:=cumprod(1+port_ret), by=.(model_key,excl_rate)]
  p_m1 <- ggplot(m1r,aes(x=date,y=cum_idx,colour=excl_rate,group=excl_rate))+
    geom_line(linewidth=0.85)+
    geom_vline(xintercept=as.numeric(as.Date("2016-01-01")),
               linetype="dashed",colour="grey40")+
    scale_colour_manual(values=RATE_COLS,labels=RATE_LABS)+
    scale_y_continuous(labels=dollar_format(prefix="$"))+
    scale_x_date(date_breaks="2 years",date_labels="%Y")+
    labs(title="M1 Overlay — Exclusion Rate Comparison (EW)",
         x=NULL,y="Portfolio Value ($1)",colour="Excl Rate")+
    theme_minimal(base_size=12)+
    theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1))
  ggsave(file.path(FIGS$index_csi_track,"m1_exclusion_rates.png"),p_m1,
         width=PLOT_WIDTH*1.2,height=PLOT_HEIGHT,dpi=PLOT_DPI)
}

cat(sprintf("\n[11_Results.R] DONE: %s\n", format(Sys.time())))