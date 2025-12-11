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



#==== 1C - Parameters =========================================================#

## Directories.
Data_Directory <- file.path(Path, "02_Data")
Data_CRSP_Directory <- file.path(Data_Directory, "CRSP")

## Date range.
start_date <- ymd("2023-01-01")
end_date <- ymd("2024-12-31")

#==== 1D - Setup WRDS =========================================================#

set_wrds_credentials()
wrds <- get_wrds_connection()

#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

#==== 02a - Read the data file ================================================#

##===========================#
## Query CRSP.
##===========================#

## Use SQL in R to query WRDS/CRSP.

# Connect to tables
msf_db <- tbl(wrds, I("crsp.msf_v2"))
stk_info_db <- tbl(wrds, I("crsp.stksecurityinfohist"))

# Define the security filter (Common Stocks, US, Major Exchanges)
# Note: These filters align with the Tidy Finance / CRSP CIZ standard

crsp_filtered <- msf_db |> 
  filter(mthcaldt >= start_date & mthcaldt <= end_date) |> 
  select(permno, mthcaldt, mthret, mthprc, shrout) |> 
  inner_join(
    stk_info_db |> 
      filter(
        sharetype == "NS" &               # Ordinary Common Shares
          securitytype == "EQTY" &          # Equity
          securitysubtype == "COM" &        # Common
          usincflg == "Y" &                 # US Incorporated
          issuertype %in% c("ACOR", "CORP") & 
          primaryexch %in% c("N", "A", "Q") & # NYSE, AMEX, NASDAQ
          conditionaltype %in% c("RW", "NW") & # Regular Way / Next Day
          tradingstatusflg == "A"           # Active
      ) |> 
      select(permno, secinfostartdt, secinfoenddt, primaryexch, siccd),
    by = "permno"
  ) |> 
  # Apply the Slowly Changing Dimension Logic (Date within validity range)
  filter(mthcaldt >= secinfostartdt & mthcaldt <= secinfoenddt) |> 
  
  # ==== ML PRE-PROCESSING (IN SQL) ==== #
  mutate(
    date = mthcaldt,
    # Fix: CRSP prices can be negative (bid/ask average). Take absolute value.
    abs_prc = abs(mthprc),
    # Calculate Market Cap (shrout is in 1000s)
    mkt_cap = abs(mthprc) * shrout * 1000
  ) |> 
  
  # Filter out micro-caps or penny stocks? (Optional but recommended for ML)
  # ML models often get noisy due to penny stocks.
  filter(abs_prc >= 1) |> 
  
  select(
    permno, date, 
    ret = mthret, 
    prc = abs_prc, 
    mkt_cap,
    shrout, primaryexch, siccd
  )

# Only collect AFTER filtering
crsp_local <- crsp_filtered |> 
  collect() |> 
  mutate(date = ymd(date)) |> 
  arrange(permno, date)

# Quick sanity check for the user
print(paste("Rows loaded:", nrow(crsp_local)))
print(paste("Unique stocks:", length(unique(crsp_local$permno))))

##===========================#
## Query Compustat.
##===========================#

# 1. Get the Link Table (The Bridge)
# We only want links that are "Primary" (P, C) and valid (LC, LU, LS)
ccm_link_db <- tbl(wrds, I("crsp.ccmxpf_lnkhist")) |>
  filter(
    linktype %in% c("LC", "LU", "LS"),
    linkprim %in% c("P", "C")
  ) |>
  select(gvkey, permno = lpermno, linkdt, linkenddt)

glimpse(ccm_link_db)

# 2. Get Compustat Fundamentals (The Accounting Data)
comp_db <- tbl(wrds, I("comp.funda"))

comp_data <- comp_db |>
  filter(
    indfmt == "INDL",    # Industrial format
    datafmt == "STD",    # Standardized format
    consol == "C",       # Consolidated
    datadate >= start_date & datadate <= end_date
  ) |>
  select(
    gvkey, datadate, fyear,
    # Balance Sheet Variables
    at, lt, seq, ceq, pstkrv, txdb, itcb,
    # Income Statement Variables
    ni, ib, sale, cogs
  ) |>
  inner_join(ccm_link_db, by = "gvkey") |> 
  # Ensure the accounting data falls within the valid link range
  filter(
    datadate >= linkdt & (is.na(linkenddt) | datadate <= linkenddt)
  ) |> 
  collect() |> 
  mutate(
    datadate = ymd(datadate),
    # Handle NA in Link End Date (NA usually means "still active")
    linkenddt = replace_na(ymd(linkenddt), ymd("2099-12-31"))
  )





#==== 02a - Read the data file ================================================#

Path <- file.path(Data_CRSP_Directory, "1926_1930.csv")
Data <- read.csv(Path)

Data <- Data %>%
  mutate(PRC = if_else(PRC < 0, NA_real_, PRC)) ## No trade occured. Is the average of bid/ask.
                                                ## Lets set it to NA for now, we have to think about how to handle it.

## Now  change to wide format.
Data$date <- as.Date(Data$date)

wide_data <- Data %>%
  select(date, CUSIP, PRC) %>%
  distinct(date, CUSIP, .keep_all = TRUE) %>% 
  pivot_wider(names_from = CUSIP, values_from = PRC) %>%
  arrange(date) %>%                       
  tidyr::fill(-date, .direction = "down")
Data_xts <- xts(wide_data[, -1], order.by = wide_data$date)


wide_data <- Data %>%
  select(date, CUSIP, PRC) %>%
  distinct(date, CUSIP, .keep_all = TRUE) %>% 
  pivot_wider(names_from = CUSIP, values_from = PRC)
Data_xts <- xts(wide_data[, -1], order.by = wide_data$date)

# View the result
head(Data_xts)
