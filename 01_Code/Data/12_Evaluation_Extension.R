#==============================================================================#
#==== 12_Index_Evaluation.R ===================================================#
#==== Index Performance Evaluation — Risk Reduction Primary, Alpha Secondary ==#
#==============================================================================#
#
# PURPOSE:
#   Comprehensive performance evaluation of all ML overlay strategies vs
#   benchmark and factor screens. Risk reduction is the primary thesis claim;
#   alpha generation is reported as secondary evidence.
#
# STRUCTURE:
#   Section 1  : Benchmark characterisation
#   Section 2  : Exclusion rate sensitivity (1%/3%/5%/10%) — all 12 models
#   Section 3  : Cross-model comparison at 5% — heatmap and bar plots
#   Section 4  : EW vs CW comparison
#   Section 5  : Period analysis (in-sample / test / OOS)
#   Section 6  : Risk reduction evidence — MaxDD, Sharpe improvement, VaR
#   Section 7  : Alpha analysis — CAPM and Fama-French regressions
#   Section 8  : ML vs factor screens (from 11B outputs)
#   Section 9  : Turnover and tracking error
#   Section 10 : Statistical significance — bootstrap Sharpe CI, t-tests
#   Section 11 : Drawdown and annual return plots
#   Section 12 : CSI avoidance analysis
#
# PRIMARY CLAIM (Section 6):
#   The overlay reduces maximum drawdown and improves risk-adjusted returns
#   (Sharpe, Sortino) vs the benchmark. Evidence: lower MaxDD, lower vol,
#   positive Sharpe delta, bootstrapped CI excludes zero.
#
# SECONDARY CLAIM (Section 7):
#   The overlay generates positive alpha vs CAPM and Fama-French.
#   Jensen's alpha t-test. Carhart 4-factor optional if momentum factor available.
#
# INPUTS:
#   config.R
#   DIR_TABLES/index_returns.rds
#   DIR_TABLES/index_performance.rds
#   DIR_TABLES/index_weights.rds
#   DIR_TABLES/index_exclusion_summary.rds
#   DIR_TABLES/quality_returns.rds
#   DIR_TABLES/quality_performance.rds
#   DIR_TABLES/quality_overlap.rds
#   DIR_MACRO/ff_factors.rds     (Fama-French 3-factor — optional)
#
# OUTPUTS:
#   DIR_TABLES/eval_risk_reduction.rds    primary claim evidence table
#   DIR_TABLES/eval_alpha.rds             regression results
#   DIR_TABLES/eval_bootstrap.rds         bootstrap Sharpe CIs
#   DIR_TABLES/eval_turnover.rds          turnover and TE
#   DIR_FIGURES/12_eval/                  all evaluation plots
#
#==============================================================================#

source("config.R")

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
  library(tidyr)
  library(lubridate)
  library(viridis)
})

cat("\n[12_Index_Evaluation.R] START:", format(Sys.time()), "\n")

## Figure directory
FIG_EVAL <- file.path(DIR_FIGURES,"12_eval")
dir.create(FIG_EVAL, recursive=TRUE, showWarnings=FALSE)

## ── Global settings ───────────────────────────────────────────────────────────
RF_ANNUAL  <- 0.03
N_BOOT     <- 1000L
ALPHA_BOOT <- 0.05
set.seed(SEED)

INSAMPLE_START <- 1998L
OOS_END        <- 2024L

PERIODS_E <- list(
  insample=c(INSAMPLE_START, year(TRAIN_END)),
  test    =c(year(TEST_START), year(TEST_END)),
  oos     =c(year(OOS_START), OOS_END),
  full    =c(INSAMPLE_START, OOS_END)
)

ggsave_e <- function(fname, p, w=PLOT_WIDTH, h=PLOT_HEIGHT) {
  path <- file.path(FIG_EVAL, fname)
  ggsave(path, p, width=w, height=h, dpi=PLOT_DPI)
  cat(sprintf("    Saved: %s\n", fname))
}

#==============================================================================#
# 0. Load inputs
#==============================================================================#

cat("[12] Loading inputs...\n")
port_returns  <- readRDS(file.path(DIR_TABLES,"index_returns.rds"))
port_perf     <- readRDS(file.path(DIR_TABLES,"index_performance.rds"))
port_weights  <- readRDS(file.path(DIR_TABLES,"index_weights.rds"))
excl_summary  <- readRDS(file.path(DIR_TABLES,"index_exclusion_summary.rds"))

qual_returns  <- if (file.exists(file.path(DIR_TABLES,"quality_returns.rds")))
  readRDS(file.path(DIR_TABLES,"quality_returns.rds")) else NULL
qual_perf     <- if (file.exists(file.path(DIR_TABLES,"quality_performance.rds")))
  readRDS(file.path(DIR_TABLES,"quality_performance.rds")) else NULL
qual_overlap  <- if (file.exists(file.path(DIR_TABLES,"quality_overlap.rds")))
  readRDS(file.path(DIR_TABLES,"quality_overlap.rds")) else NULL

## Benchmark return series (EW)
bench_ret <- port_returns[model_key=="bench" & weighting=="ew"][order(date)]

## Performance helpers
fn_perf <- function(rv, rf=RF_ANNUAL) {
  rv <- rv[is.finite(rv)]; if(length(rv)<12) return(NULL)
  ny <- length(rv)/12; rfm <- (1+rf)^(1/12)-1
  cum <- prod(1+rv)-1; cagr <- (1+cum)^(1/ny)-1
  vol <- sd(rv)*sqrt(12); exc <- rv-rfm
  sh  <- mean(exc)/sd(exc)*sqrt(12)
  ddr <- exc[rv<rfm]
  srt <- if(length(ddr)>1) mean(exc)/(sd(ddr)*sqrt(12)) else NA_real_
  ci  <- cumprod(1+rv); pk <- cummax(ci); mdd <- min((ci-pk)/pk)
  cal <- if(mdd<0) cagr/abs(mdd) else NA_real_
  var_95 <- quantile(rv, 0.05, na.rm=TRUE)
  cvar_95<- mean(rv[rv<=var_95], na.rm=TRUE)
  data.frame(n_months=length(rv),cum_ret=round(cum,4),cagr=round(cagr,4),
             vol=round(vol,4),sharpe=round(sh,4),sortino=round(srt,4),
             max_dd=round(mdd,4),calmar=round(cal,4),
             var_95=round(var_95,4),cvar_95=round(cvar_95,4))
}

cat(sprintf("  Loaded: %d strategy-months\n",nrow(port_returns)))

#==============================================================================#
# Section 1 — Benchmark characterisation
#==============================================================================#

cat("\n[12] Section 1: Benchmark characterisation...\n")

bm_stats <- lapply(names(PERIODS_E), function(pnm) {
  yr  <- PERIODS_E[[pnm]]
  sub <- bench_ret[year>=yr[1]&year<=yr[2]]
  pf  <- fn_perf(sub$port_ret); if(is.null(pf)) return(NULL)
  pf$period <- pnm; pf
})
bm_stats <- rbindlist(Filter(Negate(is.null), bm_stats), fill=TRUE)

cat("  Benchmark performance by period:\n")
print(bm_stats[,.(period,cagr,vol,sharpe,max_dd,calmar)], row.names=FALSE)

## Cumulative return with crisis annotations
bench_ret[, cum_idx := cumprod(1+port_ret)]
CRISIS <- data.frame(
  xmin=as.Date(c("2000-03-01","2007-10-01","2020-02-01")),
  xmax=as.Date(c("2002-10-01","2009-03-01","2020-04-01")),
  label=c("Dot-com","GFC","COVID")
)

p_bench_full <- ggplot(bench_ret,aes(x=date,y=cum_idx)) +
  geom_rect(data=CRISIS,aes(xmin=xmin,xmax=xmax,ymin=-Inf,ymax=Inf),
            fill="grey85",alpha=0.5,inherit.aes=FALSE) +
  geom_line(colour="#9E9E9E",linewidth=1) +
  geom_text(data=CRISIS,aes(x=(xmin+xmax)/2,y=Inf,label=label),
            vjust=1.5,size=2.8,colour="grey40",inherit.aes=FALSE) +
  geom_vline(xintercept=as.numeric(as.Date("2016-01-01")),
             linetype="dashed",colour="#1565C0") +
  geom_vline(xintercept=as.numeric(as.Date("2020-01-01")),
             linetype="dotted",colour="#E53935") +
  scale_y_continuous(labels=dollar_format(prefix="$")) +
  scale_x_date(date_breaks="2 years",date_labels="%Y") +
  labs(title="Benchmark — Pseudo-Russell 3000 (Equal-Weight)",
       subtitle="Grey bands = crisis periods | Dashed = test start | Dotted = OOS start",
       x=NULL,y="Portfolio Value ($1 invested)") +
  theme_minimal(base_size=12)

ggsave_e("s1_benchmark_full.png", p_bench_full, w=PLOT_WIDTH*1.3)

#==============================================================================#
# Section 2 — Exclusion rate sensitivity
#==============================================================================#

cat("\n[12] Section 2: Exclusion rate sensitivity...\n")

RATE_COLS <- c("none"="#9E9E9E","1pct"="#BBDEFB","3pct"="#1E88E5",
               "5pct"="#0D47A1","10pct"="#E53935")
RATE_LABS <- c("none"="Benchmark","1pct"="Excl 1%","3pct"="Excl 3%",
               "5pct"="Excl 5%","10pct"="Excl 10%")

## For each track anchor (M1/B1/S1), show rate sensitivity across OOS
for (key in intersect(c("fund","bucket","structural"), unique(port_returns$model_key))) {
  short_k <- MODEL_SHORTS[[key]] %||% key
  rt <- rbindlist(list(
    bench_ret[year>=year(OOS_START),.(date,year,port_ret,excl_rate="none")],
    port_returns[model_key==key&weighting=="ew"&year>=year(OOS_START),
                 .(date,year,port_ret,excl_rate)]
  ))
  setorder(rt,excl_rate,date)
  rt[, cum_idx := cumprod(1+port_ret), by=excl_rate]
  
  p_rate <- ggplot(rt,aes(x=date,y=cum_idx,colour=excl_rate,group=excl_rate))+
    geom_line(linewidth=0.85)+
    scale_colour_manual(values=RATE_COLS,labels=RATE_LABS)+
    scale_y_continuous(labels=dollar_format(prefix="$"))+
    scale_x_date(date_breaks="6 months",date_labels="%Y-%m")+
    labs(title=sprintf("%s — Exclusion Rate Sensitivity (EW, OOS 2020-2024)",
                       MODEL_LABELS[[key]] %||% key),
         x=NULL,y="Portfolio Value ($1)",colour="Excl Rate")+
    theme_minimal(base_size=12)+
    theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1))
  
  ggsave_e(sprintf("s2_%s_rate_sensitivity.png",short_k), p_rate, w=PLOT_WIDTH*1.1)
}

#==============================================================================#
# Section 3 — Cross-model comparison at 5% (primary comparison)
#==============================================================================#

cat("\n[12] Section 3: Cross-model comparison...\n")

## Performance heatmap: all 12 models × {Sharpe, MaxDD, CAGR} × OOS
oos5_ew <- port_perf[period=="oos" & excl_rate %in% c("none","5pct") & weighting=="ew"]
oos5_ew[, short := MODEL_SHORTS[model_key] %||% model_key]
oos5_ew[, track := MODEL_TRACK[model_key]  %||% "—"]
setorder(oos5_ew, track, short)

bench_5 <- oos5_ew[model_key=="bench"]
strat_5 <- oos5_ew[model_key!="bench"]

## Sharpe delta vs benchmark
strat_5[, sharpe_delta := sharpe - (bench_5$sharpe)]
strat_5[, maxdd_delta  := max_dd  - (bench_5$max_dd)]   ## negative = better
strat_5[, cagr_delta   := cagr    - (bench_5$cagr)]

## Bar plot: Sharpe delta
p_sh_delta <- ggplot(strat_5,aes(x=reorder(short,-sharpe_delta),
                                 y=sharpe_delta,fill=track))+
  geom_col(width=0.65,alpha=0.85)+
  geom_hline(yintercept=0,colour="black",linewidth=0.4)+
  scale_fill_manual(values=c(CSI="#1565C0",Bucket="#1B5E20",Structural="#6A1B9A"))+
  scale_y_continuous(labels=number_format(accuracy=0.001))+
  labs(title="Sharpe Ratio Delta vs Benchmark — OOS 2020-2024 (5% EW)",
       subtitle="Positive = outperforms benchmark | Blue = CSI | Green = Bucket | Purple = Structural",
       x=NULL,y="Sharpe delta",fill="Track")+
  theme_minimal(base_size=12)+theme(legend.position="bottom")
ggsave_e("s3_sharpe_delta.png", p_sh_delta)

## MaxDD delta
p_mdd_delta <- ggplot(strat_5,aes(x=reorder(short,maxdd_delta),
                                  y=maxdd_delta*100,fill=track))+
  geom_col(width=0.65,alpha=0.85)+
  geom_hline(yintercept=0,colour="black",linewidth=0.4)+
  scale_fill_manual(values=c(CSI="#1565C0",Bucket="#1B5E20",Structural="#6A1B9A"))+
  labs(title="Maximum Drawdown Delta vs Benchmark — OOS 2020-2024 (5% EW)",
       subtitle="Negative = lower drawdown than benchmark (better)",
       x=NULL,y="MaxDD delta (pp)",fill="Track")+
  theme_minimal(base_size=12)+theme(legend.position="bottom")
ggsave_e("s3_maxdd_delta.png", p_mdd_delta)

## Heatmap: metric × model
hm_dt <- melt(strat_5[,.(short,track,sharpe,max_dd,cagr,calmar,vol)],
              id.vars=c("short","track"),variable.name="metric",value.name="val")
hm_dt[, metric:=recode(metric,sharpe="Sharpe",max_dd="MaxDD",
                       cagr="CAGR",calmar="Calmar",vol="Vol")]

p_hm <- ggplot(hm_dt,aes(x=short,y=metric,fill=val))+
  geom_tile(colour="white",linewidth=0.5)+
  geom_text(aes(label=round(val,3)),size=2.8,colour="white")+
  scale_fill_viridis(option="mako",name="Value")+
  facet_wrap(~track,scales="free_x",nrow=1)+
  labs(title="Performance Heatmap — All 12 Models (OOS 2020-2024, 5% EW)",
       x=NULL,y=NULL)+
  theme_minimal(base_size=10)+
  theme(axis.text.x=element_text(angle=30,hjust=1),panel.grid=element_blank())
ggsave_e("s3_performance_heatmap.png", p_hm, w=PLOT_WIDTH*1.4, h=PLOT_HEIGHT*0.8)

#==============================================================================#
# Section 4 — EW vs CW comparison
#==============================================================================#

cat("\n[12] Section 4: EW vs CW...\n")

## M1 at 5% as the representative comparison
ew_cw <- port_perf[model_key %in% c("bench","fund") &
                     excl_rate %in% c("none","5pct") &
                     period=="oos",
                   .(model_key,weighting,excl_rate,cagr,sharpe,max_dd,calmar)]
ew_cw[, label:=sprintf("%s (%s)",
                       ifelse(model_key=="bench","Benchmark","M1 5%"),
                       toupper(weighting))]

p_ewcw <- ggplot(ew_cw,aes(x=sharpe,y=-max_dd,colour=weighting,
                           shape=model_key,label=label))+
  geom_point(size=5,alpha=0.8)+
  geom_text(nudge_y=0.005,size=3)+
  scale_colour_manual(values=c(ew="#1565C0",cw="#E53935"))+
  scale_shape_manual(values=c(bench=17,fund=16))+
  scale_y_continuous(labels=percent_format(accuracy=1))+
  labs(title="EW vs CW — Risk/Return (OOS 2020-2024, M1 5%)",
       subtitle="x = Sharpe | y = -MaxDD (higher = lower drawdown)",
       x="Sharpe Ratio",y="-MaxDD",colour="Weighting",shape="Model")+
  theme_minimal(base_size=12)+theme(legend.position="bottom")
ggsave_e("s4_ew_vs_cw.png", p_ewcw, w=PLOT_WIDTH*0.85, h=PLOT_HEIGHT)

#==============================================================================#
# Section 5 — Period analysis
#==============================================================================#

cat("\n[12] Section 5: Period breakdown...\n")

## Best 3 models by OOS Sharpe + benchmark, all periods
top3 <- oos5_ew[order(-sharpe)][1:3, model_key]
period_dt <- port_perf[model_key %in% c("bench",top3) &
                         excl_rate %in% c("none","5pct") &
                         weighting=="ew" &
                         period %in% c("insample","test","oos")]
period_dt[, label := MODEL_LABELS[model_key] %||% model_key]
period_dt[, period_label := recode(period,
                                   insample="In-sample 1998-2015",test="Test 2016-2019",oos="OOS 2020-2024")]

p_period_sharpe <- ggplot(period_dt,
                          aes(x=period_label,y=sharpe,fill=label))+
  geom_col(position=position_dodge(0.75),width=0.7,alpha=0.85)+
  geom_hline(yintercept=0,colour="black",linewidth=0.3)+
  scale_fill_viridis(discrete=TRUE,option="mako")+
  labs(title="Sharpe Ratio by Period — Benchmark vs Top 3 Models (5% EW)",
       x=NULL,y="Sharpe Ratio",fill=NULL)+
  theme_minimal(base_size=11)+theme(legend.position="bottom")
ggsave_e("s5_sharpe_by_period.png", p_period_sharpe, w=PLOT_WIDTH*1.1)

#==============================================================================#
# Section 6 — Risk reduction evidence (PRIMARY CLAIM)
#==============================================================================#

cat("\n[12] Section 6: Risk reduction evidence...\n")

## For each model at 5% EW, compute risk metrics vs benchmark (OOS)
risk_rows <- list()
for (key in unique(port_returns[model_key!="bench",model_key])) {
  for (wt in c("ew","cw")) {
    yr  <- PERIODS_E$oos
    r_s <- port_returns[model_key==key&excl_rate=="5pct"&
                          weighting==wt&year>=yr[1]&year<=yr[2],port_ret]
    r_b <- bench_ret[year>=yr[1]&year<=yr[2],port_ret]
    if(length(r_s)<12||length(r_b)<12) next
    pf_s <- fn_perf(r_s); pf_b <- fn_perf(r_b)
    if(is.null(pf_s)||is.null(pf_b)) next
    
    risk_rows[[paste0(key,wt)]] <- data.frame(
      model_key    = key,
      weighting    = wt,
      short        = MODEL_SHORTS[[key]] %||% key,
      track        = MODEL_TRACK[[key]]  %||% "—",
      cagr_strat   = pf_s$cagr,     cagr_bench  = pf_b$cagr,
      sharpe_strat = pf_s$sharpe,   sharpe_bench = pf_b$sharpe,
      maxdd_strat  = pf_s$max_dd,   maxdd_bench  = pf_b$max_dd,
      vol_strat    = pf_s$vol,      vol_bench    = pf_b$vol,
      var_strat    = pf_s$var_95,   var_bench    = pf_b$var_95,
      cvar_strat   = pf_s$cvar_95,  cvar_bench   = pf_b$cvar_95,
      cagr_delta   = pf_s$cagr   - pf_b$cagr,
      sharpe_delta = pf_s$sharpe - pf_b$sharpe,
      maxdd_delta  = pf_s$max_dd - pf_b$max_dd,
      vol_delta    = pf_s$vol    - pf_b$vol,
      var_delta    = pf_s$var_95 - pf_b$var_95,
      stringsAsFactors=FALSE
    )
  }
}
risk_table <- rbindlist(risk_rows,fill=TRUE)
saveRDS(risk_table, file.path(DIR_TABLES,"eval_risk_reduction.rds"))

## Risk reduction summary table
cat("\n  PRIMARY CLAIM — Risk Reduction (OOS, EW, 5%):\n")
rt_ew <- risk_table[weighting=="ew"]
setorder(rt_ew,track,short)
cat(sprintf("  %-8s %-12s %8s %8s %9s %8s\n",
            "Model","Track","SharpeΔ","MaxDDΔ","CAGRΔ","VaRΔ"))
cat(paste(rep("-",55),collapse=""),"\n")
for(j in seq_len(nrow(rt_ew))) {
  r <- rt_ew[j]
  cat(sprintf("  %-8s %-12s %+8.3f %+7.2f%% %+8.2f%% %+7.2f%%\n",
              r$short,r$track,r$sharpe_delta,r$maxdd_delta*100,
              r$cagr_delta*100,r$var_delta*100))
}

n_improved_sharpe <- sum(rt_ew$sharpe_delta>0)
n_improved_maxdd  <- sum(rt_ew$maxdd_delta<0)
cat(sprintf("\n  Summary: %d/%d models improve Sharpe | %d/%d reduce MaxDD\n",
            n_improved_sharpe,nrow(rt_ew),n_improved_maxdd,nrow(rt_ew)))

#==============================================================================#
# Section 7 — Alpha analysis (CAPM + Fama-French)
#==============================================================================#

cat("\n[12] Section 7: Alpha analysis...\n")

## Load Fama-French factors if available
ff_path <- file.path(DIR_MACRO, "ff_factors.rds")
if (file.exists(ff_path)) {
  ff      <- as.data.table(readRDS(ff_path))
  has_ff  <- TRUE
  cat("  Fama-French factors loaded.\n")
} else {
  cat("  FF factors not found — CAPM only.\n")
  has_ff <- FALSE
}

## CAPM regression: excess_return_strategy ~ alpha + beta * excess_return_mkt
## Fama-French: excess_return_strategy ~ alpha + beta*MktRf + s*SMB + h*HML

alpha_rows <- list()
yr_oos <- PERIODS_E$oos

for (key in unique(port_returns[model_key!="bench",model_key])) {
  r_s  <- port_returns[model_key==key&excl_rate=="5pct"&
                         weighting=="ew"&year>=yr_oos[1]&year<=yr_oos[2],
                       .(date,year,month,port_ret)]
  r_b  <- bench_ret[year>=yr_oos[1]&year<=yr_oos[2],
                    .(date,port_ret_bench=port_ret)]
  dat  <- merge(r_s,r_b,by="date")
  if(nrow(dat)<12) next
  
  rfm  <- (1+RF_ANNUAL)^(1/12)-1
  dat[, excess_s := port_ret - rfm]
  dat[, excess_b := port_ret_bench - rfm]
  
  ## CAPM
  capm <- tryCatch(lm(excess_s ~ excess_b, data=dat), error=function(e) NULL)
  if (!is.null(capm)) {
    sc  <- summary(capm)$coefficients
    alpha_rows[[paste0(key,"_capm")]] <- data.frame(
      model_key=key, short=MODEL_SHORTS[[key]]%||%key,
      track=MODEL_TRACK[[key]]%||%"—",
      model="CAPM", period="oos",
      alpha_monthly=round(sc[1,1],5),
      alpha_tstat  =round(sc[1,3],3),
      alpha_pval   =round(sc[1,4],4),
      beta         =round(sc[2,1],4),
      r_squared    =round(summary(capm)$r.squared,4),
      alpha_annual =round(sc[1,1]*12,4),
      stringsAsFactors=FALSE
    )
  }
  
  ## Fama-French 3-factor
  if (has_ff) {
    dat_ff <- merge(dat, ff[,.(date,MktRf,SMB,HML,RF)], by="date", all.x=TRUE)
    dat_ff <- dat_ff[!is.na(MktRf)&!is.na(SMB)&!is.na(HML)]
    if (nrow(dat_ff)>=12) {
      dat_ff[, excess_ff := port_ret - RF]
      ff3 <- tryCatch(
        lm(excess_ff ~ MktRf + SMB + HML, data=dat_ff),
        error=function(e) NULL)
      if (!is.null(ff3)) {
        sc3 <- summary(ff3)$coefficients
        alpha_rows[[paste0(key,"_ff3")]] <- data.frame(
          model_key=key, short=MODEL_SHORTS[[key]]%||%key,
          track=MODEL_TRACK[[key]]%||%"—",
          model="FF3", period="oos",
          alpha_monthly=round(sc3[1,1],5),
          alpha_tstat  =round(sc3[1,3],3),
          alpha_pval   =round(sc3[1,4],4),
          beta         =round(sc3[2,1],4),
          r_squared    =round(summary(ff3)$r.squared,4),
          alpha_annual =round(sc3[1,1]*12,4),
          stringsAsFactors=FALSE
        )
      }
    }
  }
}

alpha_table <- rbindlist(alpha_rows,fill=TRUE)
saveRDS(alpha_table, file.path(DIR_TABLES,"eval_alpha.rds"))

cat("\n  Alpha table (CAPM, OOS, 5% EW):\n")
capm_t <- alpha_table[model=="CAPM"]
setorder(capm_t,track,short)
cat(sprintf("  %-8s %-12s %10s %8s %8s %8s\n",
            "Model","Track","α/yr","t-stat","p-val","R²"))
for(j in seq_len(nrow(capm_t))) {
  r <- capm_t[j]
  sig <- if(!is.na(r$alpha_pval))
    ifelse(r$alpha_pval<0.01,"***",ifelse(r$alpha_pval<0.05,"**",
                                          ifelse(r$alpha_pval<0.10,"*",""))) else ""
  cat(sprintf("  %-8s %-12s %+9.2f%% %8.3f %8.4f%s %7.4f\n",
              r$short,r$track,r$alpha_annual*100,r$alpha_tstat,r$alpha_pval,sig,r$r_squared))
}

## Alpha bar plot
if (nrow(alpha_table)>0) {
  p_alpha <- ggplot(alpha_table[model=="CAPM"],
                    aes(x=reorder(short,-alpha_annual),
                        y=alpha_annual*100,fill=track))+
    geom_col(width=0.65,alpha=0.85)+
    geom_hline(yintercept=0,colour="black",linewidth=0.4)+
    geom_text(aes(label=sprintf("%.2f%%",alpha_annual*100)),
              vjust=ifelse(alpha_table[model=="CAPM"]$alpha_annual>0,-0.3,1.3),
              size=2.8)+
    scale_fill_manual(values=c(CSI="#1565C0",Bucket="#1B5E20",Structural="#6A1B9A"))+
    scale_y_continuous(labels=percent_format(accuracy=0.1))+
    labs(title="CAPM Alpha (Annualised) — OOS 2020-2024 (5% EW)",
         subtitle="* p<0.10 | ** p<0.05 | *** p<0.01",
         x=NULL,y="Annual Alpha",fill="Track")+
    theme_minimal(base_size=11)+theme(legend.position="bottom")
  ggsave_e("s7_capm_alpha.png", p_alpha)
}

#==============================================================================#
# Section 8 — ML vs factor screens
#==============================================================================#

cat("\n[12] Section 8: ML vs factor comparison...\n")

if (!is.null(qual_returns) && !is.null(qual_perf)) {
  
  ## Side-by-side OOS cumulative (M1 + all factor overlays)
  m1_oos <- port_returns[model_key=="fund"&excl_rate=="5pct"&
                           weighting=="ew"&year>=year(OOS_START),
                         .(date,year,port_ret,label="M1 ML (5%)",grp="ml")]
  fac_oos <- qual_returns[strategy=="overlay"&weighting=="ew"&year>=year(OOS_START),
                          .(date,year,port_ret,label=paste0(toupper(substr(factor,1,1)),
                                                            substr(factor,2,nchar(factor))," Screen"),
                            grp="factor")]
  b_oos   <- qual_returns[factor=="bench"&weighting=="ew"&year>=year(OOS_START),
                          .(date,year,port_ret,label="Benchmark",grp="bench")]
  
  comp_oos <- rbindlist(list(m1_oos,fac_oos,b_oos),use.names=TRUE,fill=TRUE)
  setorder(comp_oos,label,date)
  comp_oos[, cum_idx:=cumprod(1+port_ret), by=label]
  
  COMP8_COLS <- c("M1 ML (5%)"="#1565C0","Benchmark"="#9E9E9E",
                  "Lowvol Screen"="#FF9800","Quality Screen"="#1B5E20",
                  "Momentum Screen"="#9C27B0")
  
  p_ml_fac <- ggplot(comp_oos,aes(x=date,y=cum_idx,colour=label,group=label))+
    geom_line(linewidth=0.9)+
    scale_colour_manual(values=COMP8_COLS)+
    scale_y_continuous(labels=dollar_format(prefix="$"))+
    scale_x_date(date_breaks="6 months",date_labels="%Y-%m")+
    labs(title="ML Overlay vs Factor Screens — OOS 2020-2024 (EW, 5%)",
         x=NULL,y="Portfolio Value ($1)",colour=NULL)+
    theme_minimal(base_size=12)+
    theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1))
  ggsave_e("s8_ml_vs_factor_oos.png", p_ml_fac, w=PLOT_WIDTH*1.1)
  
  ## Jaccard heatmap
  if (!is.null(qual_overlap) && nrow(qual_overlap)>0) {
    qual_overlap_dt <- as.data.table(qual_overlap)
    qual_overlap_dt[, ml_label:=MODEL_LABELS[ml_model]%||%ml_model]
    p_jacc <- ggplot(qual_overlap_dt,
                     aes(x=factor,y=ml_label,fill=jaccard))+
      geom_tile(colour="white",linewidth=0.5)+
      geom_text(aes(label=round(jaccard,3)),size=3.5,colour="white")+
      scale_fill_viridis(option="mako",limits=c(0,1),name="Jaccard")+
      labs(title="Exclusion Overlap: ML Models vs Factor Screens",
           subtitle="1.0 = identical exclusion sets | 0.0 = no overlap",
           x="Factor Screen",y="ML Model")+
      theme_minimal(base_size=12)
    ggsave_e("s8_jaccard_heatmap.png", p_jacc, h=PLOT_HEIGHT*0.85)
  }
}

#==============================================================================#
# Section 9 — Turnover and tracking error
#==============================================================================#

cat("\n[12] Section 9: Turnover and tracking error...\n")

## Tracking error vs benchmark (OOS, 5%, EW)
te_rows <- list()
yr_te <- PERIODS_E$oos

for (key in unique(port_returns[model_key!="bench",model_key])) {
  r_s <- port_returns[model_key==key&excl_rate=="5pct"&
                        weighting=="ew"&year>=yr_te[1]&year<=yr_te[2],
                      .(date,port_ret)]
  r_b <- bench_ret[year>=yr_te[1]&year<=yr_te[2],.(date,port_ret)]
  if(nrow(r_s)<12) next
  m   <- merge(r_s,r_b,by="date",suffixes=c("_s","_b"))
  m[, active_ret := port_ret_s - port_ret_b]
  te  <- sd(m$active_ret)*sqrt(12)
  ir  <- mean(m$active_ret)/sd(m$active_ret)*sqrt(12)
  
  te_rows[[key]] <- data.frame(
    model_key=key, short=MODEL_SHORTS[[key]]%||%key,
    track=MODEL_TRACK[[key]]%||%"—",
    tracking_error=round(te,4),
    info_ratio=round(ir,4),
    mean_active=round(mean(m$active_ret)*12,4),
    stringsAsFactors=FALSE
  )
}
te_table <- rbindlist(te_rows,fill=TRUE)
saveRDS(te_table, file.path(DIR_TABLES,"eval_turnover.rds"))

cat("\n  Tracking error (OOS, 5% EW):\n")
setorder(te_table,track,short)
print(te_table[,.(short,track,tracking_error,info_ratio,mean_active)],row.names=FALSE)

## TE vs IR scatter
if (nrow(te_table)>0) {
  p_te <- ggplot(te_table,aes(x=tracking_error*100,y=info_ratio,
                              colour=track,label=short))+
    geom_point(size=4)+
    geom_text(nudge_y=0.05,size=3)+
    geom_hline(yintercept=0,linetype="dashed",colour="grey50")+
    scale_colour_manual(values=c(CSI="#1565C0",Bucket="#1B5E20",Structural="#6A1B9A"))+
    labs(title="Tracking Error vs Information Ratio (OOS, 5% EW)",
         subtitle="IR = mean active return / TE | right-upper = best",
         x="Annualised Tracking Error (%)",y="Information Ratio",colour="Track")+
    theme_minimal(base_size=12)+theme(legend.position="bottom")
  ggsave_e("s9_te_vs_ir.png", p_te, w=PLOT_WIDTH*0.9)
}

#==============================================================================#
# Section 10 — Bootstrap Sharpe CI + paired t-tests
#==============================================================================#

cat("\n[12] Section 10: Bootstrap Sharpe CI and t-tests...\n")

fn_sharpe_boot <- function(rv, rf=RF_ANNUAL, n=N_BOOT, a=ALPHA_BOOT) {
  rv   <- rv[is.finite(rv)]
  rfm  <- (1+rf)^(1/12)-1
  exc  <- rv-rfm
  obs  <- replicate(n, {
    idx <- sample(length(exc),replace=TRUE)
    mean(exc[idx])/sd(exc[idx])*sqrt(12)
  })
  c(sharpe=mean(exc)/sd(exc)*sqrt(12),
    lo=quantile(obs,a/2), hi=quantile(obs,1-a/2))
}

boot_rows <- list()
yr_bt <- PERIODS_E$oos

## Benchmark CI
rb <- bench_ret[year>=yr_bt[1]&year<=yr_bt[2],port_ret]
cb <- fn_sharpe_boot(rb)
boot_rows[["bench"]] <- data.frame(model_key="bench",short="BENCH",
                                   track="—",sharpe=cb["sharpe"],lo=cb["lo"],hi=cb["hi"],
                                   sig_positive=cb["lo"]>0, stringsAsFactors=FALSE)

for (key in unique(port_returns[model_key!="bench",model_key])) {
  rs <- port_returns[model_key==key&excl_rate=="5pct"&
                       weighting=="ew"&year>=yr_bt[1]&year<=yr_bt[2],port_ret]
  if(length(rs)<24) next
  cs <- fn_sharpe_boot(rs)
  boot_rows[[key]] <- data.frame(
    model_key=key, short=MODEL_SHORTS[[key]]%||%key,
    track=MODEL_TRACK[[key]]%||%"—",
    sharpe=cs["sharpe"], lo=cs["lo"], hi=cs["hi"],
    sig_positive=cs["lo"]>0,
    stringsAsFactors=FALSE
  )
}
boot_table <- rbindlist(boot_rows,fill=TRUE)
saveRDS(boot_table, file.path(DIR_TABLES,"eval_bootstrap.rds"))

## Bootstrap CI plot
p_boot <- ggplot(boot_table,aes(x=reorder(short,-sharpe),
                                y=sharpe,ymin=lo,ymax=hi,
                                colour=track))+
  geom_pointrange(size=0.5)+
  geom_hline(yintercept=0,linetype="dashed",colour="grey50")+
  scale_colour_manual(values=c("—"="#9E9E9E",CSI="#1565C0",
                               Bucket="#1B5E20",Structural="#6A1B9A"))+
  labs(title=sprintf("Sharpe Ratio with Bootstrap 95%% CI — OOS (5%% EW, n=%d resamples)",N_BOOT),
       x=NULL,y="Sharpe Ratio",colour="Track")+
  theme_minimal(base_size=11)+
  theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1))
ggsave_e("s10_bootstrap_sharpe.png", p_boot, w=PLOT_WIDTH*1.3)

## Paired t-test: strategy excess returns vs benchmark (OOS, 5%, EW)
ttest_rows <- list()
for (key in unique(port_returns[model_key!="bench",model_key])) {
  rs <- port_returns[model_key==key&excl_rate=="5pct"&
                       weighting=="ew"&year>=yr_bt[1]&year<=yr_bt[2],
                     .(date,port_ret)]
  rb2<- bench_ret[year>=yr_bt[1]&year<=yr_bt[2],.(date,port_ret)]
  m  <- merge(rs,rb2,by="date",suffixes=c("_s","_b"))
  m[, diff:=port_ret_s-port_ret_b]
  if(nrow(m)<12) next
  tt <- t.test(m$diff, mu=0, alternative="two.sided")
  ttest_rows[[key]] <- data.frame(
    model_key=key, short=MODEL_SHORTS[[key]]%||%key,
    track=MODEL_TRACK[[key]]%||%"—",
    mean_diff_ann=round(mean(m$diff)*12,4),
    t_stat=round(tt$statistic,3),
    p_val =round(tt$p.value,4),
    sig   =ifelse(tt$p.value<0.01,"***",ifelse(tt$p.value<0.05,"**",
                                               ifelse(tt$p.value<0.10,"*",""))),
    stringsAsFactors=FALSE
  )
}
ttest_table <- rbindlist(ttest_rows,fill=TRUE)

cat("\n  Paired t-test — strategy vs benchmark (OOS, 5% EW):\n")
setorder(ttest_table,track,short)
print(ttest_table[,.(short,track,mean_diff_ann,t_stat,p_val,sig)],row.names=FALSE)

#==============================================================================#
# Section 11 — Drawdown and annual return plots
#==============================================================================#

cat("\n[12] Section 11: Drawdown and annual return plots...\n")

## Track anchors at 5% EW — full period
anchor_keys <- intersect(c("fund","bucket","structural"),
                         unique(port_returns$model_key))

all_dt <- rbindlist(list(
  bench_ret[,.(date,year,month,port_ret,model_key="bench",excl_rate="none",weighting="ew")],
  port_returns[model_key %in% anchor_keys & excl_rate=="5pct" & weighting=="ew",
               .(date,year,month,port_ret,model_key,excl_rate,weighting)]
), use.names=TRUE, fill=TRUE)
setorder(all_dt,model_key,date)
all_dt[, cum_idx := cumprod(1+port_ret), by=model_key]
all_dt[, pk      := cummax(cum_idx),     by=model_key]
all_dt[, dd      := (cum_idx-pk)/pk]

ANCH_COLS <- c(bench="#9E9E9E",fund="#1565C0",bucket="#1B5E20",structural="#6A1B9A")
ANCH_LABS <- c(bench="Benchmark",fund="M1 (CSI)",bucket="B1 (Bucket)",structural="S1 (Structural)")

## Cumulative return — full
p_cum_full <- ggplot(all_dt,aes(x=date,y=cum_idx,colour=model_key,group=model_key))+
  geom_rect(data=CRISIS,aes(xmin=xmin,xmax=xmax,ymin=-Inf,ymax=Inf),
            fill="grey88",alpha=0.4,inherit.aes=FALSE)+
  geom_line(linewidth=0.9)+
  geom_vline(xintercept=as.numeric(as.Date("2016-01-01")),
             linetype="dashed",colour="grey40")+
  geom_vline(xintercept=as.numeric(as.Date("2020-01-01")),
             linetype="dotted",colour="grey40")+
  scale_colour_manual(values=ANCH_COLS,labels=ANCH_LABS)+
  scale_y_continuous(labels=dollar_format(prefix="$"))+
  scale_x_date(date_breaks="2 years",date_labels="%Y")+
  labs(title="Crash-Filtered Index — Full Period (5% EW)",
       subtitle="Grey = crisis periods | Dashed = test 2016 | Dotted = OOS 2020",
       x=NULL,y="Portfolio Value ($1)",colour=NULL)+
  theme_minimal(base_size=12)+
  theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1))
ggsave_e("s11_cumulative_full.png", p_cum_full, w=PLOT_WIDTH*1.3)

## Drawdown
p_dd <- ggplot(all_dt,aes(x=date,y=dd*100,colour=model_key,group=model_key))+
  geom_rect(data=CRISIS,aes(xmin=xmin,xmax=xmax,ymin=-Inf,ymax=Inf),
            fill="grey88",alpha=0.4,inherit.aes=FALSE)+
  geom_line(linewidth=0.85)+
  geom_hline(yintercept=0,linewidth=0.3)+
  scale_colour_manual(values=ANCH_COLS,labels=ANCH_LABS)+
  scale_y_continuous(labels=function(x) paste0(x,"%"))+
  scale_x_date(date_breaks="2 years",date_labels="%Y")+
  labs(title="Drawdown — Benchmark vs Track Anchors (5% EW)",
       subtitle="Lower drawdown = primary thesis claim",
       x=NULL,y="Drawdown (%)",colour=NULL)+
  theme_minimal(base_size=12)+
  theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1))
ggsave_e("s11_drawdown.png", p_dd, w=PLOT_WIDTH*1.3)

## Annual returns bar chart
ann_ret <- all_dt[,.(ann_ret=prod(1+port_ret)-1),by=.(model_key,year)]
ann_ret[, label:=ANCH_LABS[model_key]]

p_ann <- ggplot(ann_ret,aes(x=year,y=ann_ret*100,fill=model_key))+
  geom_col(position=position_dodge(0.8),width=0.7,alpha=0.85)+
  geom_hline(yintercept=0,colour="black",linewidth=0.3)+
  geom_vline(xintercept=2015.5,linetype="dashed",colour="grey40")+
  geom_vline(xintercept=2019.5,linetype="dotted",colour="grey40")+
  scale_fill_manual(values=ANCH_COLS,labels=ANCH_LABS)+
  scale_y_continuous(labels=function(x) paste0(x,"%"))+
  labs(title="Annual Returns — Benchmark vs Track Anchors (5% EW)",
       x=NULL,y="Annual Return (%)",fill=NULL)+
  theme_minimal(base_size=10)+
  theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1,size=8))
ggsave_e("s11_annual_returns.png", p_ann, w=PLOT_WIDTH*1.5, h=PLOT_HEIGHT)

#==============================================================================#
# Section 12 — Exclusion diagnostics
#==============================================================================#

cat("\n[12] Section 12: Exclusion diagnostics...\n")

## Firms excluded per year by rate and track (December)
excl_ann <- excl_summary[weighting=="ew" & q_year >= INSAMPLE_START]
excl_ann[, track := MODEL_TRACK[model_key] %||% "—"]
excl_ann[, short := MODEL_SHORTS[model_key] %||% model_key]

## Plot for each exclusion rate — anchor models only
anchor_excl <- excl_ann[model_key %in% c(anchor_keys,"bench")]

p_excl <- ggplot(anchor_excl[excl_rate!="none"],
                 aes(x=q_year,y=excl_pct,colour=model_key,group=model_key))+
  geom_line(linewidth=0.9)+
  facet_wrap(~excl_rate,nrow=2)+
  geom_vline(xintercept=year(TEST_START)-0.5,linetype="dashed",colour="grey40")+
  geom_vline(xintercept=year(OOS_START)-0.5,linetype="dotted",colour="grey40")+
  scale_colour_manual(values=ANCH_COLS,labels=ANCH_LABS)+
  scale_y_continuous(labels=function(x) paste0(x,"%"))+
  labs(title="Actual Exclusion Rate by Year — Anchor Models",
       subtitle="Dashed = test 2016 | Dotted = OOS 2020 | Rank-based: should be flat",
       x="Year",y="% Excluded",colour=NULL)+
  theme_minimal(base_size=11)+theme(legend.position="bottom")
ggsave_e("s12_exclusion_rate_by_year.png", p_excl, w=PLOT_WIDTH*1.2, h=PLOT_HEIGHT*1.2)

#==============================================================================#
# Final summary
#==============================================================================#

cat("\n[12] ══════════════════════════════════════════════════════\n")
cat("  EVALUATION SUMMARY\n")
cat("  ══════════════════════════════════════════════════════\n\n")

cat("  PRIMARY CLAIM — Risk Reduction (OOS 2020-2024, 5% EW):\n")
cat(sprintf("  %d/%d models improve Sharpe vs benchmark\n",
            sum(risk_table[weighting=="ew",sharpe_delta]>0,na.rm=TRUE),
            nrow(risk_table[weighting=="ew"])))
cat(sprintf("  %d/%d models reduce MaxDD vs benchmark\n",
            sum(risk_table[weighting=="ew",maxdd_delta]<0,na.rm=TRUE),
            nrow(risk_table[weighting=="ew"])))

cat("\n  SECONDARY CLAIM — Alpha Generation (CAPM, OOS, 5% EW):\n")
if(nrow(alpha_table)>0) {
  capm_pos <- alpha_table[model=="CAPM"&alpha_annual>0]
  capm_sig <- alpha_table[model=="CAPM"&alpha_pval<0.10]
  cat(sprintf("  %d/%d models generate positive annual alpha\n",
              nrow(capm_pos),nrow(alpha_table[model=="CAPM"])))
  cat(sprintf("  %d/%d models significant at 10%% level\n",
              nrow(capm_sig),nrow(alpha_table[model=="CAPM"])))
}

cat("\n  FACTOR ORTHOGONALITY:\n")
if (!is.null(qual_overlap)) {
  qo <- as.data.table(qual_overlap)
  cat(sprintf("  Mean Jaccard overlap (ML vs factor screens): %.3f\n",
              mean(qo$jaccard,na.rm=TRUE)))
  cat(sprintf("  %s\n",
              if(mean(qo$jaccard,na.rm=TRUE)<0.30)
                "Low overlap — ML overlay captures information beyond simple factor screens"
              else
                "Moderate overlap — partial substitutability with factor screens"))
}

cat(sprintf("\n[12_Index_Evaluation.R] DONE: %s\n", format(Sys.time())))