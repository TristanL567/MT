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

Functions_Directory <- file.path(Directory, "01_Code/Data/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Date range.
start_date <- ymd("2023-01-01")
end_date <- ymd("2024-12-31")

## Saved objects.

# Data_Stock_Masterlist.rds - Masterlist of all firms after filtering.
# Data_Monthly - Contains all data about the firms in the masterlist.
# Data_Failed_Companies - Contains the tibble with the failed company details.
# Data_y - Final tibble of firms with the target and if they are in a zombie state (target is then NA).

#==== 1D - Setup WRDS =========================================================#

Path_Environment <- file.path(Directory, ".Renviron")
readRenviron(Path_Environment)
Sys.getenv("WRDS_USER")
# set_wrds_credentials()
wrds <- get_wrds_connection()

print(wrds)

##=========================================================================#
## 2. Setup & Universe Selection (Constituents)
##=========================================================================#

##========================##
## Connect to the WRDS / CRSP database.
##========================##

msf_db <- tbl(wrds, I("crsp.msf"))
stk_info_db <- tbl(wrds, I("crsp.stksecurityinfohist"))

colnames(msf_db)

### define that firms need to be active.
max_db_date <- stk_info_db |>
  summarise(max_dt = max(secinfoenddt, na.rm = TRUE)) |>
  pull(max_dt)

##========================##
## Data aggregation.
##========================##

lifetime_stats <- stk_info_db |>
  group_by(permno) |>
  summarise(
    listing_date = min(secinfostartdt, na.rm = TRUE),
    removal_date = max(secinfoenddt, na.rm = TRUE),
    n_changes = n() # Proxy for corporate complexity
  ) |>
  ungroup()

### Get the firm specific information.
final_attributes <- stk_info_db |>
  group_by(permno) |>
  filter(secinfoenddt == max(secinfoenddt, na.rm = TRUE)) |>
  ungroup() |>
  select(
    permno,
    ticker, 
    issuernm, 
    primaryexch,
    securitytype, 
    securitysubtype, 
    sharetype, 
    shareclass,
    siccd, 
    naics,
    delactiontype, 
    delreasontype
  )

##========================##
## Get the overview.
##========================##

Data_Stock_Masterlist_Raw <- lifetime_stats |>
  inner_join(final_attributes, by = "permno") |>
  collect() |> # Execute SQL and pull to R
  mutate(
    is_active = if_else(removal_date >= max_db_date, "Yes", "No"),
    lifetime_years = round(as.numeric(removal_date - listing_date) / 365.25, 2)
  ) |>
  select(
    permno, ticker, issuernm, is_active, 
    listing_date, removal_date, lifetime_years,
    delactiontype, delreasontype,
    securitytype, securitysubtype, sharetype, shareclass,
    siccd, naics, primaryexch, n_changes
  ) |>
  arrange(permno)

Path <- file.path(Data_CRSP_Directory, "Data_Stock_Masterlist_Raw.rds")
saveRDS(Data_Stock_Masterlist_Raw, file = Path)

# 5. Preview the Master List
glimpse(Data_Stock_Masterlist_Raw)

## Dimensions.

dim(Data_Stock_Masterlist_Raw)
unique(Data_Stock_Masterlist_Raw$securitytype)

## 6A. Filter for EQUITY (and NA) only.
stock_master_list_filtered <- Data_Stock_Masterlist_Raw |>
  filter(
    !securitytype %in% c("FUND"),
    removal_date > "1966-01-01",
    lifetime_years > 4.99,
    primaryexch %in% c("N", "A", "Q") # NYSE, AMEX, NASDAQ
  )

## 6B. filter for companies with two share classes, but same company characteristics.
Data_Stock_Masterlist <- stock_master_list_filtered |>
  group_by(issuernm) |>
  arrange(desc(lifetime_years), permno) |>
  slice(1) |>
  ungroup()

summary(Data_Stock_Masterlist)

## Save to the data directory.

Path <- file.path(Data_CRSP_Directory, "Data_Stock_Masterlist.rds")
saveRDS(Data_Stock_Masterlist, file = Path)

# Data_Stock_Masterlist_Raw <- readRDS(Path)

##=========================================================================#
## 3. Retrieve the constiuent characteristics.
##=========================================================================#

# Filter for the "Investable Universe"
# We do this first to avoid pulling data for dead penny stocks we won't trade.
universe_subset <- Data_Stock_Masterlist |>
  select(permno, listing_date, removal_date)

##=========================================================================#
## 4. Stock Prices & Returns (Monthly)
##=========================================================================#

##========================##
## Download the firm returns.
##========================##

target_permnos <- Data_Stock_Masterlist_Raw$permno

Data_Monthly_Raw <- msf_db |>
  filter(
    permno %in% target_permnos,
    date >= start_date,  # Note: Standard MSF usually uses 'date', not 'mthcaldt'
    date <= end_date
  ) |>
  select(
    permno, 
    date, 
    prc,      # Price (Check for negative values later)
    ret,      # Total Return
    retx,     # Return without Dividends
    shrout,   # Shares Outstanding
    vol       # Volume
  ) |>
  collect()

##========================##
## Compute returns (total return and gross dividends).
##========================##

Data_Monthly <- Data_Monthly_Raw |>
  arrange(permno, date) |>
  mutate(
    price_close = abs(prc),
    ret_total = if_else(ret < -1, NA_real_, ret),
    ret_price = if_else(retx < -1, NA_real_, retx),
    div_yield = ret_total - ret_price,
    prev_price = lag(price_close),
    div_amount = div_yield * prev_price,
    market_cap = price_close * shrout
  ) |>
  filter(!is.na(ret_total)) |>
  select(
    permno, 
    date, 
    price = price_close, 
    tot_ret_net_dvds = ret_total, # Total return (includes dividends)
    price_ret = ret_price,        # Price return (excludes dividends)
    div_amount,                   # Calculated gross dividend amount
    market_cap,
    vol,
    shrout
  )

Path <- file.path(Data_CRSP_Directory, "Data_Monthly.rds")
saveRDS(Data_Monthly, file = Path)

##=========================================================================#
## 5. Check for catastrophic implosions (same as the Tewari et al. Methodology)
##=========================================================================#

PARAM_C <- -0.8   # Crash threshold (Drawdown <= -80%)
PARAM_M <- -0.2   # Recovery ceiling (Wealth cannot recover above -20% of Peak)
PARAM_T <- 18     # Zombie Period: 18 months

##========================##
## Compute the drawdowns.
##========================##

Data_Signal_Monthly <- Data_Monthly |>
  group_by(permno) |>
  arrange(date) |>
  mutate(
    safe_ret = replace_na(tot_ret_net_dvds, 0),
    wealth_index = cumprod(1 + safe_ret),
    running_max_wealth = cummax(wealth_index),
    drawdown = (wealth_index / running_max_wealth) - 1,
    is_crash_zone = drawdown <= PARAM_C
  ) |>
  ungroup()

##========================##
## Implosion universe.
##========================##

implosion_universe <- Data_Signal_Monthly |>
  group_by(permno) |>
  mutate(
    recovery_ceiling = running_max_wealth * (1 + PARAM_M),
    
    future_max_wealth = zoo::rollapply(wealth_index, width = PARAM_T, 
                                       FUN = max, 
                                       align = "left", 
                                       fill = NA)
  ) |>
  ungroup() |>
  filter(
    is_crash_zone,
    future_max_wealth <= recovery_ceiling
  )

##========================##
## Universe of the failed companies.
##========================##

Data_Failed_Companies <- implosion_universe |>
  group_by(permno) |>
  summarise(
    implosion_date = min(date),
    wealth_at_failure = first(wealth_index[date == min(date)]),
    peak_wealth = first(running_max_wealth[date == min(date)]),
    price_at_failure = first(price[date == min(date)]), 
    lifetime_drawdown = first(drawdown[date == min(date)])
  ) |>
  ungroup()

# Preview results
print(paste("Identified", nrow(Data_Failed_Companies), "failed companies based on Total Returns."))
head(Data_Failed_Companies)

Path <- file.path(Data_CRSP_Directory, "Data_Failed_Companies.rds")
saveRDS(Data_Failed_Companies, file = Path)

##=========================================================================#
## 6. Setup the data in the long-format.
##=========================================================================#

#### rework the 9999.

PREDICTION_HORIZON <- 12  # Predict failure 12 months ahead
BUFFER_WINDOW      <- 6   # Gap between Safe and Danger (Months 13-18)
ZOMBIE_WINDOW      <- 18  # Post-crash exclusion

model_universe <- Data_Monthly |>
  left_join(Data_Failed_Companies |> select(permno, implosion_date), 
            by = "permno") |>
  mutate(
    months_dist = interval(date, implosion_date) %/% months(1),
    months_dist = replace_na(months_dist, 9999)
  )

##========================##
## Create the final tibble.
##========================##

Data_y <- model_universe |>
  mutate(
    y = case_when(
      # 1. DANGER ZONE (0 to 12 months before failure)
      months_dist >= 0 & months_dist <= PREDICTION_HORIZON ~ 1,
      
      # 2. BUFFER ZONE (13 to 18 months before failure - Ambiguous)
      months_dist > PREDICTION_HORIZON & months_dist <= (PREDICTION_HORIZON + BUFFER_WINDOW) ~ NA_real_,
      
      # 3. ZOMBIE ZONE (The crash itself + 18 months after)
      # We remove this period as the firm is "dead" or extremely volatile.
      months_dist < 0 & months_dist >= -ZOMBIE_WINDOW ~ NA_real_,
      
      # 4. RECOVERY ZONE (More than 18 months post-crash)
      # FIX: We explicitly label these as 0 (Safe) to preserve them.
      months_dist < -ZOMBIE_WINDOW ~ 0,
      
      # 5. SAFE ZONE (Healthy firms or long before failure)
      TRUE ~ 0
    )
  ) |>
  filter(!is.na(y)) |>
  select(
    permno, 
    date, 
    market_cap,    
    price, 
    ret = tot_ret_net_dvds, 
    y, 
    months_dist
  )

##========================##
## Preview and output.
##========================##

table(Data_y$y)

Path <- file.path(Data_CRSP_Directory, "Data_y.rds")
saveRDS(Data_y, file = Path)

##=========================================================================#
## 7. Descriptive statistics of the dataset.
##=========================================================================#