#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#


#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Directory <- file.path(here::here("")) ## You need to install the package first incase you do not have it.

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

packages <- c("here", "xts", "dplyr", "tidyr",
              "RPostgres", "tidyverse", "tidyfinance", "scales",
              "RSQLite", "dbplyr", "lubridate"
)

for(i in 1:length(packages)){
  package_name <- packages[i]
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name, character.only = TRUE)
    cat(paste("Package '", package_name, "' was not installed. It has now been installed and loaded.\n", sep = ""))
  } else {
    cat(paste("Package '", package_name, "' is already installed and has been loaded.\n", sep = ""))
  }
  library(package_name, character.only = TRUE)
}

#==== 1B - Functions ==========================================================#

sourceFunctions <- function (functionDirectory)  {
  functionFiles <- list.files(path = functionDirectory, pattern = "*.R", 
                              full.names = T)
  ssource <- function(path) {
    try(source(path, echo = F, verbose = F, print.eval = T, 
               local = F))
  }
  sapply(functionFiles, ssource)
}

#==== 1C - Parameters =========================================================#

## Directories.
Data_Directory <- file.path(Directory, "02_Data")
Data_CRSP_Directory <- file.path(Data_Directory, "CRSP")
Data_Compustat_Directory <- file.path(Data_Directory, "Compustat")
Data_Dataset_Directory <- file.path(Data_Directory, "Dataset")

Functions_Directory <- file.path(Directory, "01_Code/Data/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Date range.
start_date <- ymd("1965-01-01")
end_date <- ymd("2024-12-31")

##=============================================================================#
##==== 2 - Input Data =========================================================#
##=============================================================================#

##==== 2A - Input Data =========================================================#

##=================================##
## Load the dependent variable data.
##=================================##

Path <- file.path(Data_CRSP_Directory, "Data_y.rds")
Data_y <- readRDS(Path)

##=================================##
## Check for delistings.
##=================================##

Path <- file.path(Data_CRSP_Directory, "Annual_Event_Check.rds")
Annual_Event_Check <- readRDS(Path)

##=================================##
## Load the features (X variable space).
##=================================##

Path <- file.path(Data_Compustat_Directory, "Data_Compustat_Annual.rds")
Data_Compustat_Annual <- readRDS(Path)

##==== 2B - Clean Compustat Data ==============================================#

##=================================##
## Clean the compustat data.
##=================================##

Data_Compustat_Clean <- Data_Compustat_Annual |>
  arrange(gvkey, datadate) |>
  group_by(gvkey) |>
  fill(sich, .direction = "updown") |>
  
  # --- 1. Impute Zeros ---
  mutate(
    # Balance Sheet
    invt   = replace_na(invt, 0),
    intan  = replace_na(intan, 0),
    gdwl   = replace_na(gdwl, 0),
    txdba  = replace_na(txdba, 0),
    wcap   = replace_na(wcap, 0),
    
    dd1    = replace_na(dd1, 0),
    dlc    = replace_na(dlc, 0),
    dltt   = replace_na(dltt, 0),
    drc    = replace_na(drc, 0),
    txp    = replace_na(txp, 0),
    txditc = replace_na(txditc, 0),
    
    pstk   = replace_na(pstk, 0),
    mib    = replace_na(mib, 0),
    
    # Income Statement
    xrd    = replace_na(xrd, 0),
    xad    = replace_na(xad, 0),
    xrent  = replace_na(xrent, 0),
    spi    = replace_na(spi, 0),
    nopi   = replace_na(nopi, 0),
    fca    = replace_na(fca, 0),
    dp     = replace_na(dp, 0),
    
    xinst  = replace_na(xinst, 0),
    idit   = replace_na(idit, 0),
    dv     = replace_na(dv, 0),
    dvt    = replace_na(dvt, 0)
    
    # Other: Removed dpr
  ) |>
  
  # --- 2. Market Data ---
  mutate(
    mkt_cap = mkvalt,
    price_abs = abs(prcc_f),
    price_adj = price_abs / ajex
  ) |>
  
  # --- 3. Filter ---
  filter(at > 0, sale > 0, !is.na(mkt_cap), !is.na(sich)) |>
  ungroup() |>
  
  # --- 4. Select ---
  select(
    permno, gvkey, datadate, fyear, sich,
    
    # Balance Sheet
    at, act, che, ivst, rect, invt, aco, wcap,
    ppent, ivpt, intan, gdwl, txdba, ao,
    lt, lct, ap, txp, drc, dlc, dd1,
    dltt, txditc, lo,
    seq, mib, cstk, caps, pstk, re, acominc, tstk,
    
    # Income Statement
    sale, cogs, gp,
    xsga, xrd, xad, xrent, dp, spi,
    ebit,
    xint, xinst, idit, nopi, fca,
    pi, txt, xi, ni,
    epsfi, ci, dv, dvt,
    
    # Other (Removed dpr)
    emp, oibdp, ppegt,
    mkt_cap, mkvalt, prcc_f, csho, ajex, price_abs, price_adj
  ) |>
  arrange(permno, datadate)

# Preview
glimpse(Data_Compustat_Clean)

##=============================================================================#
##==== 3 - Merging ============================================================#
##=============================================================================#

##==== 3A - Merge the two datasets ============================================#

Data_Targets_Annual <- Data_y |>
  mutate(year = year(date)) |>
  filter(month(date) == 12) |>
  arrange(permno, year, desc(date)) |>
  group_by(permno, year) |>
  slice(1) |>
  ungroup() |>
  transmute(
    permno,
    pred_date = date,            
    pred_year = year,      
    target_next_12m = y,
    trading_status = trading_status
  )

###
glimpse(Data_Targets_Annual)

##=================================##
## Merge the datasets (assuming no delay in financial data release).
##=================================##

Dataset <- Data_Targets_Annual |>
  left_join(
    Data_Compustat_Clean, 
    by = join_by(permno, pred_year == fyear)
  ) |>
  left_join(Annual_Event_Check, by = c("permno", "pred_year" = "year")) |>
  mutate(
    dataset_type = "Naive (NoDelay)",
    trading_status = case_when(
      was_halted_or_delisted ~ "Trading Halted / No Status",
      is.na(trading_status) ~ "Active (Major Exchange)",
      TRUE ~ trading_status
    )
  ) |>
  select(permno, pred_date, target_next_12m, pred_year, trading_status, everything(), -was_halted_or_delisted)

# 
# Dataset <- Data_Targets_Annual |>
#   left_join(
#     Data_Compustat_Clean, 
#     by = join_by(permno, pred_year == fyear)
#   ) |>
#   mutate(dataset_type = "Naive (NoDelay)") |>
#   select(permno, pred_date, target_next_12m, pred_year, trading_status, everything())

## Now compute the time to the last failed event.

Dataset <- Dataset |>
  arrange(permno, pred_year) |>
  group_by(permno) |>
  mutate(
    implosion_start_year = ifelse(coalesce(target_next_12m, 0) == 1, pred_year, NA_real_),
    last_event_year = lag(implosion_start_year)
  ) |>
  fill(last_event_year, .direction = "down") |>
  mutate(
    years_since_last_implosion = pred_year - last_event_year,
    was_zombie_ever = !is.na(years_since_last_implosion),
    years_since_last_implosion = replace_na(years_since_last_implosion, -1)
  ) |>
  ungroup() |>
  relocate(years_since_last_implosion, was_zombie_ever, .after = pred_year)

##=================================##
## Remove unecessary columns.
##=================================##

Dataset <- Dataset |>
  rename(y = target_next_12m) |>
  filter(!is.na(at)) |>
  filter(!is.na(y)) |>
  select(
    -implosion_start_year, 
    -last_event_year
  )

### Check the dataset if we have observations where there is a 1 in the next year too.
Dataset <- Dataset |>
  arrange(permno, pred_date) |>  # <--- Sorted by permno and pred_date
  group_by(permno) |>
  mutate(
    prev_y = lag(y, default = 0),
    prev_year = lag(pred_year, default = 0)
  ) |>
  # Remove consecutive '1's (if year diff is exactly 1)
  filter(!(y == 1 & prev_y == 1 & (pred_year - prev_year == 1))) |>
  select(-prev_y, -prev_year) |>
  ungroup()

### Check the number of CSI events.
Events_Per_Firm <- Dataset |>
  group_by(permno) |>
  summarise(
    total_events = sum(y, na.rm = TRUE),
    min_year = min(pred_year),
    max_year = max(pred_year)
  ) |>
  arrange(desc(total_events))

##=================================##
## Merge the datasets (assuming a 3 month delay in financial data release).
##=================================##

# Data_Features_Realistic <- Data_Compustat_Clean |>
#   mutate(public_date = datadate + months(3))


##=============================================================================#
##==== 4 - Output =============================================================#
##=============================================================================#

##==== 4 - Output =============================================================#

##=================================##
## Save the Dataset (without assuming a data delay).
##=================================##

Path <- file.path(Data_Dataset_Directory, "Dataset.rds")
saveRDS(Dataset, file = Path)

##=================================##
## Save the Dataset (assuming a 3-month data delay).
##=================================##

# Path <- file.path(Data_Dataset_Directory, "Dataset_Delayed.rds")
# saveRDS(Dataset_Delayed, file = Path)

##=============================================================================#
##=============================================================================#
##=============================================================================#