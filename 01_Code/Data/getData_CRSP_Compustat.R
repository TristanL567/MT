#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#


#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

packages <- c("here", "xts", "dplyr", "tidyr",
              "RPostgres", "tidyverse", "tidyfinance", "scales",
              "RSQLite", "dbplyr"
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
Data_Directory <- file.path(Path, "02_Data")
Data_CRSP_Directory <- file.path(Data_Directory, "CRSP")

Functions_Directory <- file.path(Path, "01_Code/Data/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Date range.
start_date <- ymd("2023-01-01")
end_date <- ymd("2024-12-31")

#==== 1D - Setup WRDS =========================================================#

Path_Environment <- file.path(Path, ".Renviron")
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
stock_master_list <- lifetime_stats |>
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

# 5. Preview the Master List
glimpse(stock_master_list)

## Dimensions.

dim(stock_master_list)
unique(stock_master_list$securitytype)

## 6A. Filter for EQUITY (and NA) only.
stock_master_list_filtered <- stock_master_list |>
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

stock_master_list_filtered <- stock_master_list_filtered |>
  group_by(issuernm) |>
  arrange(desc(lifetime_years), permno) |>
  slice(1) |>
  ungroup()

summary(stock_master_list_filtered)

##=========================================================================#
## 3. Retrieve the constiuent characteristics.
##=========================================================================#

# Filter for the "Investable Universe"
# We do this first to avoid pulling data for dead penny stocks we won't trade.
universe_subset <- stock_master_list_filtered |>
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
crsp_monthly <- raw_prices |>
  inner_join(universe_subset, by = "permno") |>
  filter(mthcaldt >= "1960-01-01") |>
  filter(mthcaldt >= listing_date & mthcaldt <= removal_date) |>
    mutate(
    date = ymd(mthcaldt),
    prc = abs(mthprc), 
    mthret = if_else(is.na(mthret), 0, mthret),
    log_ret = if_else(1 + mthret > 0, log(1 + mthret), NA_real_),
    mkt_cap = prc * shrout * 1000 
  ) |>
  select(permno, date, ret = mthret, log_ret, prc, mkt_cap) |>
  arrange(permno, date) |>
  filter(!is.na(prc))

# Preview
glimpse(crsp_monthly)

##=========================================================================#
## 5. Check for catastrophic implosions (same as the Tewari et al. Methodology)
##=========================================================================#

# 1. Define Parameters (from Appendix)
PARAM_C <- -0.8   # Crash threshold (Drawdown <= -80%)
PARAM_M <- -0.2   # Recovery ceiling (Price cannot recover above -20% of Peak)
PARAM_T <- 18     # Zombie Period: 78 weeks approx 18 months

# 2. Calculate Peak and Drawdown
monthly_signals <- crsp_monthly |>
  group_by(permno) |>
  arrange(date) |>
  mutate(
    running_max_price = cummax(prc),
    drawdown = (prc / running_max_price) - 1,
    is_crash_zone = drawdown <= PARAM_C
  ) |>
  ungroup()

# 3. Verify the "Zombie" Condition (Look-Ahead)
implosion_universe <- monthly_signals |>
  group_by(permno) |>
  mutate(
    recovery_ceiling = running_max_price * (1 + PARAM_M),
      future_max_price = zoo::rollapply(prc, width = PARAM_T, 
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
failed_companies <- implosion_universe |>
  group_by(permno) |>
  summarise(
    implosion_date = min(date),
    peak_price = first(running_max_price[date == min(date)]),
    crash_price = first(prc[date == min(date)]),
    lifetime_drawdown = first(drawdown[date == min(date)])
  ) |>
  ungroup()

# Preview results
print(paste("Identified", nrow(failed_companies), "failed companies."))
head(failed_companies)

##=========================================================================#
## 6. Setup the data in the long-format.
##=========================================================================#

# 1. Define Modeling Horizons (in Months)
PREDICTION_HORIZON <- 12  # We want to predict failure 12 months ahead
ZOMBIE_WINDOW      <- 18  # The "dead" period defined in your appendix (78 weeks)

# 2. Join Monthly Data with Failure Dates
# We keep all historical months for all stocks (failed or not)
model_universe <- crsp_monthly |>
  left_join(failed_companies |> select(permno, implosion_date), 
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
ml_panel <- model_universe |>
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
  select(permno, date, mkt_cap, prc, ret, TARGET_12M, is_zombie, months_dist, implosion_date)

##=========================================================================#
## 7. Descriptive statistics of the dataset.
##=========================================================================#












##=========================================================================#
## 5. Balance Sheet with EXACT Timestamps (Quarterly)
##=========================================================================#
# Note: specific timestamps (RDQ) are found in Quarterly (fundq), not Annual (funda).
# This is crucial for distress prediction to capture the "news" moment.

comp_q_db <- tbl(wrds, I("comp.fundq"))
ccm_link_db <- tbl(wrds, I("crsp.ccmxpf_lnkhist")) # The Link Table

# 5a. Filter Link Table
valid_links <- ccm_link_db |>
  filter(linktype %in% c("LC", "LU", "LS"), linkprim %in% c("P", "C")) |>
  select(gvkey, permno = lpermno, linkdt, linkenddt)

# 5b. Fetch Fundamentals
fund_data <- comp_q_db |>
  filter(
    indfmt == "INDL", datafmt == "STD", consol == "C",
    datadate >= start_date # Broad filter first
  ) |>
  select(
    gvkey, datadate, fyearq, fqtr,
    rdq,            # << THE KEY: Report Date of Quarterly Earnings
    atq, ltq,       # Assets, Liabilities (Quarterly)
    cheq,           # Cash & Equivalents
    niq,            # Net Income (Quarterly)
    saleq           # Sales (Quarterly)
  ) |>
  inner_join(valid_links, by = "gvkey") |>
  filter(
    datadate >= linkdt & (is.na(linkenddt) | datadate <= linkenddt)
  ) |>
  collect() |>
  mutate(
    datadate = ymd(datadate),
    rdq = ymd(rdq),
    # If RDQ is missing, we must impute a "Safe Lag" (standard is +3 or +4 months for quarters)
    public_date = case_when(
      !is.na(rdq) ~ rdq,
      TRUE ~ datadate + months(3) # Fallback if no report date exists
    )
  ) |>
  # Filter: Ensure we only keep data that was actually released
  filter(!is.na(public_date)) |>
  select(permno, gvkey, datadate, public_date, atq, ltq, cheq, niq)

# View the result: exact "public_date" tells you when the algorithm can "see" this data.
print(head(fund_data))