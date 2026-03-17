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
              "RSQLite", "dbplyr", "lubridate", "data.table",
              "slider"
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
start_date <- ymd("1960-01-01")
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
## Optimized Download
##========================##

## Pulls ALL the data within this date range. Be careful. High time complexity.
Data_Monthly_Raw <- msf_db |>
  filter(
    date >= start_date,
    date <= end_date
  ) |>
  select(
    permno, 
    date, 
    prc,      # Price 
    ret,      # Total Return
    retx,     # Return without Dividends
    shrout,   # Shares Outstanding
    vol       # Volume
  ) |>
  collect() # Bring to R Memory

Path <- file.path(Data_CRSP_Directory, "Data_Monthly_Raw.rds")
saveRDS(Data_Monthly_Raw, file = Path)

## Now we filter for the permno we actually want.
target_permnos <- Data_Stock_Masterlist$permno

Data_Monthly_Raw_Filtered <- Data_Monthly_Raw |>
  filter(permno %in% target_permnos)

##========================##
## Check for delistings.
##========================##

Names_Events <- tbl(wrds, sql("SELECT * FROM crsp.msenames"))
Delisting_Events <- tbl(wrds, sql("SELECT * FROM crsp.msedelist"))

Names_Local <- Names_Events |> 
  select(permno, namedt, nameendt, exchcd, ticker) |> 
  collect()

Delisting_Local <- Delisting_Events |> 
  select(permno, dlstdt, dlstcd) |> 
  collect()

Data_Monthly_Complete <- Data_Monthly_Raw_Filtered |>
  left_join(Names_Local, by = "permno") |>
  filter(date >= namedt & date <= nameendt) |>
  left_join(Delisting_Local, by = c("permno", "date" = "dlstdt")) |>
  mutate(
    trading_status = case_when(
      dlstcd == 100 ~ "Active (Major Exchange)",
      !is.na(dlstcd) ~ paste0("Delisted: Code ", dlstcd),
      exchcd %in% c(1, 2, 3) ~ "Active (Major Exchange)",
      exchcd == 0 ~ "Trading Halted / No Status",
      
      TRUE ~ "Inactive/OTC/Unknown"
    )
  )

Path <- file.path(Data_CRSP_Directory, "Data_Monthly_Complete.rds")
saveRDS(Data_Monthly_Complete, file = Path)

# Check the result
count(Data_Monthly_Complete, trading_status)

#### Check for delistings.
Annual_Event_Check <- Data_Monthly_Complete |>
  mutate(year = year(date)) |>
  group_by(permno, year) |>
  summarise(
    # Check if ANY month in this year had a "Delisted" or "Halted" status
    # Note: 'Delisted: Code 100' was already cleaned to "Active" in the previous step,
    # so str_detect(..., "Delisted") only catches bad delistings.
    was_halted_or_delisted = any(
      str_detect(trading_status, "Delisted") | 
        trading_status == "Trading Halted / No Status"
    ),
    .groups = "drop"
  )

Path <- file.path(Data_CRSP_Directory, "Annual_Event_Check.rds")
saveRDS(Annual_Event_Check, file = Path)

##========================##
## 2. Compute Returns
##========================##

Data_Monthly <- Data_Monthly_Complete |>
  filter(is.na(ret) | ret > -1) |> 
  arrange(permno, date) |>
  group_by(permno) |> 
  mutate(
    price_close = abs(prc),
    ret_total = ret,
    ret_price = retx,
    prev_price = lag(price_close),
    div_yield = ret_total - ret_price,
    div_amount = div_yield * prev_price,
    market_cap = price_close * shrout
  ) |>
  ungroup() |> 
  filter(!is.na(ret_total)) |>
  select(
    permno, 
    date, 
    price = price_close, 
    tot_ret_net_dvds = ret_total, 
    price_ret = ret_price, 
    div_amount, 
    market_cap,
    vol,
    shrout,
    trading_status
  )

Path <- file.path(Data_CRSP_Directory, "Data_Monthly.rds")
saveRDS(Data_Monthly, file = Path)

##=========================================================================#
## 5. Check for catastrophic implosions (same as the Tewari et al. Methodology)
##=========================================================================#

Path <- file.path(Data_CRSP_Directory, "Data_Monthly.rds")
Data_Monthly <- readRDS(Path)



PARAM_C <- -0.8   # Crash threshold
PARAM_M <- -0.2   # Recovery ceiling
PARAM_T <- 18     # Zombie Period (months)

### Compute the wealth index.
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

### Get all potential implosion events.
potential_implosions <- Data_Signal_Monthly |>
  group_by(permno) |>
  mutate(
    recovery_ceiling = running_max_wealth * (1 + PARAM_M),
    future_max_wealth = zoo::rollapply(wealth_index, width = PARAM_T, 
                                       FUN = max, align = "left", fill = NA)
  ) |>
  filter(
    is_crash_zone,
    future_max_wealth <= recovery_ceiling
  ) |>
  select(permno, date) |>
  arrange(permno, date)

### Filtering for the implosion events. (Needs to be rewored).
filter_distinct_events <- function(dates, zombie_months = 18) {
  # Handle empty or single cases
  if(length(dates) == 0) return(as.Date(character(0)))
  if(length(dates) == 1) return(dates)
  
  # Ensure sorted order is guaranteed
  dates <- sort(dates)
  
  valid_dates <- dates[1]
  last_valid <- dates[1]
  
  # FIX: Iterate by index to preserve Date class
  for(i in 2:length(dates)) {
    current_date <- dates[i]
    
    # Calculate difference in days
    # (zombie_months * 30.44) approximates the months to days
    if(as.numeric(current_date - last_valid) > (zombie_months * 30.44)) {
      valid_dates <- c(valid_dates, current_date)
      last_valid <- current_date
    }
  }
  return(valid_dates)
}

Data_Failed_Events <- potential_implosions |>
  group_by(permno) |>
  reframe(implosion_date = filter_distinct_events(date, PARAM_T)) |>
  ungroup()

print(paste("Identified", nrow(Data_Failed_Events), "distinct implosion events."))

Path <- file.path(Data_CRSP_Directory, "Data_Failed_Events.rds")
saveRDS(Data_Failed_Events, file = Path)

##=========================================================================#
## 6. Setup the data in the long-format.
##=========================================================================#

PREDICTION_HORIZON <- 12
BUFFER_WINDOW      <- 6
ZOMBIE_WINDOW      <- 18

# Convert to data.table objects (Reference copies, very fast)
dt_monthly <- as.data.table(Data_Monthly)
dt_events  <- as.data.table(Data_Failed_Events)

# Ensure dates are Date objects
dt_monthly[, date := as.Date(date)]
dt_events[, implosion_date := as.Date(implosion_date)]

# Initialize the target column y as 0 (Safe/Normal)
dt_monthly[, y := 0]

dt_events[, `:=`(
  start_target = implosion_date - (PREDICTION_HORIZON * 30.4375),
  end_target   = implosion_date
)]

dt_monthly[dt_events, 
           on = .(permno, date >= start_target, date <= end_target), 
           y := 1]

dt_events[, `:=`(
  start_buffer = implosion_date - ((PREDICTION_HORIZON + BUFFER_WINDOW) * 30.4375),
  end_buffer   = implosion_date - (PREDICTION_HORIZON * 30.4375) - 1 # Avoid overlap
)]

dt_monthly[dt_events, 
           on = .(permno, date >= start_buffer, date <= end_buffer), 
           y := NA_real_]

dt_events[, `:=`(
  start_zombie = implosion_date + 1,
  end_zombie   = implosion_date + (ZOMBIE_WINDOW * 30.4375)
)]

dt_monthly[dt_events, 
           on = .(permno, date >= start_zombie, date <= end_zombie), 
           y := NA_real_]

# Convert back to tibble/df
Data_y <- as_tibble(dt_monthly)


#### New methodology:
Data_y <- Drawdown_classified

##========================##
## Compute the annualized returns for each firm.
##========================##

Data_y_annualized <- Data_y |>
  mutate(year = year(date)) |>
  group_by(permno, year) |>
  summarise(
    annual_ret = prod(1 + tot_ret_net_dvds, na.rm = TRUE) - 1,
    n_months = n(),
    .groups = "drop"
  )

##========================##
## Preview and output.
##========================##

table(Data_y$y, useNA = "always")
length(unique(Data_y$permno))

### Now output.
Path <- file.path(Data_CRSP_Directory, "Data_y.rds")
saveRDS(Data_y, file = Path)

### Now output.
Path <- file.path(Data_CRSP_Directory, "Data_y_annualized.rds")
saveRDS(Data_y_annualized, file = Path)

##=========================================================================#
##=========================================================================#
##=========================================================================#