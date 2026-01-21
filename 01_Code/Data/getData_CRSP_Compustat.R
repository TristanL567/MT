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

# Connect to WRDS (Assuming 'wrds' connection object exists)
msf_db <- tbl(wrds, I("crsp.msf_v2"))
stk_info_db <- tbl(wrds, I("crsp.stksecurityinfohist"))

# 1. Define the end of the dataset to determine "Active" status
# We find the maximum date across the entire database.
max_db_date <- stk_info_db |>
  summarise(max_dt = max(secinfoenddt, na.rm = TRUE)) |>
  pull(max_dt)

# 2. Aggregation: Get Lifetime Stats (Start and End Dates)
# We group by 'permno' to find when the stock was first born and when it died.
lifetime_stats <- stk_info_db |>
  group_by(permno) |>
  summarise(
    listing_date = min(secinfostartdt, na.rm = TRUE),
    removal_date = max(secinfoenddt, na.rm = TRUE),
    n_changes = n() # Proxy for corporate complexity
  ) |>
  ungroup()

# 3. Selection: Get the "Final State" Attributes
# We grab the attributes from the row corresponding to the stock's *last day*.
final_attributes <- stk_info_db |>
  # Filter to keep only the row where the end date matches the stock's max end date
  group_by(permno) |>
  filter(secinfoenddt == max(secinfoenddt, na.rm = TRUE)) |>
  ungroup() |>
  select(
    permno,
    ticker, 
    issuernm, 
    primaryexch,
    # Classification columns you requested
    securitytype, 
    securitysubtype, 
    sharetype, 
    shareclass,
    # Industry Codes
    siccd, 
    naics,
    # The "Implosion" Labels (Crucial for your research)
    delactiontype, 
    delreasontype
  )

# 4. Merge and Finalize
Data_Stock_Masterlist_Raw <- lifetime_stats |>
  inner_join(final_attributes, by = "permno") |>
  collect() |> # Execute SQL and pull to R
  mutate(
    # Determine Active Status
    is_active = if_else(removal_date >= max_db_date, "Yes", "No"),
    
    # Calculate Duration
    lifetime_years = round(as.numeric(removal_date - listing_date) / 365.25, 2)
  ) |>
  # Reorder columns for readability
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

## 6B. filter for any remaining ETFs or mutual funds (not necessary if filtered for the exchanges before).
# Filter_Output <- filterFunds(Input = stock_master_list_filtered)
# stock_master_list_filtered <- Filter_Output[[""]]
# removed <- Filter_Output[["removed"]]

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

# 1. Extract the list of target Permnos
target_permnos <- universe_subset$permno

# 2. Retrieve Raw Data from WRDS (Server Side)
raw_prices <- msf_db |>
  filter(permno %in% target_permnos) |> 
  select(permno, mthcaldt, mthret, mthprc, shrout) |>
  collect()

# 3. Merge & Filter (Local R Side)
Data_Monthly <- raw_prices |>
  inner_join(universe_subset, by = "permno") |>
  filter(mthcaldt >= "1960-01-01") |>
  filter(mthcaldt >= listing_date & mthcaldt <= removal_date) |>
    mutate(
    date = ymd(mthcaldt),
    price = abs(mthprc), 
    ret = if_else(is.na(mthret), 0, mthret),
    log_ret = if_else(1 + mthret > 0, log(1 + mthret), NA_real_),
    mkt_cap = price * shrout * 1000 
  ) |>
  select(permno, date, ret = mthret, log_ret, price, mkt_cap) |>
  arrange(permno, date) |>
  filter(!is.na(price))

# Preview
# glimpse(Data_Monthly)
# length(unique(Data_Monthly$permno))

Path <- file.path(Data_CRSP_Directory, "Data_Monthly.rds")
saveRDS(Data_Monthly, file = Path)

##=========================================================================#
## 5. Check for catastrophic implosions (same as the Tewari et al. Methodology)
##=========================================================================#

# 1. Define Parameters (from Appendix)
PARAM_C <- -0.8   # Crash threshold (Drawdown <= -80%)
PARAM_M <- -0.2   # Recovery ceiling (Price cannot recover above -20% of Peak)
PARAM_T <- 18     # Zombie Period: 78 weeks approx 18 months

# 2. Calculate Peak and Drawdown
Data_Signal_Monthly <- Data_Monthly |>
  group_by(permno) |>
  arrange(date) |>
  mutate(
    running_max_price = cummax(price),
    drawdown = (price / running_max_price) - 1,
    is_crash_zone = drawdown <= PARAM_C
  ) |>
  ungroup()

# 3. Verify the "Zombie" Condition (Look-Ahead)
implosion_universe <- Data_Signal_Monthly |>
  group_by(permno) |>
  mutate(
    recovery_ceiling = running_max_price * (1 + PARAM_M),
      future_max_price = zoo::rollapply(price, width = PARAM_T, 
                                      FUN = max, 
                                      align = "left", 
                                      fill = NA)
  ) |>
  ungroup() |>
  filter(
    # Criteria 1: The Crash (Drawdown < -80%)
    is_crash_zone,
    
    # Criteria 2: The Zombie Phase (Next 18m max price < Recovery Ceiling)
    future_max_price <= recovery_ceiling
  )

# 4. Final List of Failed Companies
# A stock might flag as "failed" for 50 months in a row. 
# We only want the FIRST date it happened (the specific moment of failure).
Data_Failed_Companies <- implosion_universe |>
  group_by(permno) |>
  summarise(
    implosion_date = min(date),
    peak_price = first(running_max_price[date == min(date)]),
    crash_price = first(price[date == min(date)]),
    lifetime_drawdown = first(drawdown[date == min(date)])
  ) |>
  ungroup()

# Preview results
print(paste("Identified", nrow(Data_Failed_Companies), "failed companies."))
head(Data_Failed_Companies)

Path <- file.path(Data_CRSP_Directory, "Data_Failed_Companies.rds")
saveRDS(Data_Failed_Companies, file = Path)

##=========================================================================#
## 6. Setup the data in the long-format.
##=========================================================================#

# 1. Define Modeling Horizons (in Months)
PREDICTION_HORIZON <- 12  # We want to predict failure 12 months ahead
ZOMBIE_WINDOW      <- 18  # The "dead" period defined in your appendix (78 weeks)

# 2. Join Monthly Data with Failure Dates
# We keep all historical months for all stocks (failed or not)
model_universe <- Data_Monthly |>
  left_join(Data_Failed_Companies |> select(permno, implosion_date), 
            by = "permno") |>
  mutate(
    # A. Calculate Time Distance to Failure (in Months)
    # Negative values mean the failure is in the past.
    # Positive values mean the failure is in the future.
    months_dist = interval(date, implosion_date) %/% months(1),
    # Handle non-failed companies (infinite distance)
    months_dist = ifelse(is.na(months_dist), 9999, months_dist)
  )

# 3. Apply the "Recovery" & "Target" Logic
Data_y <- model_universe |>
  mutate(
    # --- A. The Zombie Dummy (State Flag) ---
    # Active if date is AFTER implosion but WITHIN the 18-month window
    is_zombie = if_else(months_dist < 0 & months_dist >= -ZOMBIE_WINDOW, 1, 0),
    
    # --- B. The Prediction Target (Y) ---
    TARGET_12M = case_when(
      # 1. Danger Zone: Failure is 1-12 months away
      months_dist >= 0 & months_dist <= PREDICTION_HORIZON ~ 1,
      
      # 2. Zombie Zone: Currently crashing. 
      # We set Target to NA because predicting "will I crash?" is invalid 
      # when you are currently crashing.
      is_zombie == 1 ~ NA_real_,
      
      # 3. Safe/Survivor: Healthy or Recovered (>18m post-crash)
      TRUE ~ 0
    )
  ) |>
  # Note: We do NOT filter out NA targets yet. We keep the rows.
  select(permno, date, mkt_cap, price, ret, TARGET_12M, is_zombie, months_dist, implosion_date)

# Count how many unique firms have at least one positive target (1)
unique_failed_firms <- ml_panel |>
  filter(TARGET_12M == 1) |>
  summarise(
    n_firms = n_distinct(permno),  # The number of unique companies
    total_obs = n()                # The total number of monthly rows labeled "1"
  )

print(unique_failed_firms)

# Check for firms with > 1 distinct implosion date
multiple_crash_check <- ml_panel |>
  filter(!is.na(implosion_date)) |>   # Ignore healthy companies (NA date)
  group_by(permno) |>
  summarise(
    unique_dates = n_distinct(implosion_date)
  ) |>
  filter(unique_dates > 1)

# View results (Should be empty)
if(nrow(multiple_crash_check) == 0) {
  print("Validation Passed: Every firm has exactly one unique implosion date.")
} else {
  print("Warning: The following firms have multiple implosion dates:")
  print(multiple_crash_check)
}

#### Save the data.

Path <- file.path(Data_CRSP_Directory, "Data_y.rds")
saveRDS(Data_y, file = Path)

##=========================================================================#
## 7. Descriptive statistics of the dataset.
##=========================================================================#