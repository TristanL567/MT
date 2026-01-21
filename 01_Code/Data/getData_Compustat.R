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

Functions_Directory <- file.path(Directory, "01_Code/Data/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Date range.
start_date <- ymd("1965-01-01")
end_date <- ymd("2024-12-31")

#==== 1D - Setup WRDS =========================================================#

Path_Environment <- file.path(Directory, ".Renviron")
readRenviron(Path_Environment)
Sys.getenv("WRDS_USER")
# set_wrds_credentials()
wrds <- get_wrds_connection()

print(wrds)

##=============================================================================#
##==== 2 - Input Data =========================================================#
##=============================================================================#

##==== 2A - Input Data =========================================================#

Path <- file.path(Data_CRSP_Directory, "Data_y.rds")
Data_y <- readRDS(Path)

##=================================##
## Define the permno id vector.
##=================================##

id <- unique(Data_y$permno)

##=============================================================================#
##==== 3 - Compustat Data =====================================================#
##=============================================================================#

##==== 3 - Data Download ======================================================#

##=================================##
## Setup the database connection.
##=================================##

comp_a_db <- tbl(wrds, I("comp.funda"))
ccm_link_db <- tbl(wrds, I("crsp.ccmxpf_lnkhist"))

##=================================##
## First filter the link table.
##=================================##

valid_links <- ccm_link_db |>
  filter(
    lpermno %in% target_permnos, 
    linktype %in% c("LC", "LU", "LS"), 
    linkprim %in% c("P", "C")
  ) |>
  select(gvkey, permno = lpermno, linkdt, linkenddt)

##=================================##
## Fetch the fundamental data.
##=================================##

Data_Compustat_Annual_Raw <- comp_a_db |>
  inner_join(valid_links, by = "gvkey") |> 
  filter(
    indfmt == "INDL", datafmt == "STD", consol == "C",
    datadate >= start_date, 
    datadate >= linkdt & (is.na(linkenddt) | datadate <= linkenddt)
  ) |>
  select(
    # --- Identifiers & Meta ---
    permno, gvkey, datadate, fyear, sich,
    
    # --- 1. Core Performance ("The Truth") ---
    sale,       # Net Sales
    gp,         # Gross Profit (Pricing Power / Inflation Pass-through)
    cogs,       # Cost of Goods Sold
    xsga,       # SG&A (Operational Overhead)
    xrd,        # R&D Expense (Burn Rate - Crucial for Biotech/Tech)
    ni,         # Net Income
    ebit,       # Earnings Before Interest & Taxes
    oancf,      # Operating Cash Flow (Paper: Top Predictor) [cite: 55]
    
    # --- 2. Debt Structure (Interest Rate Interactions) ---
    dltt,       # Long-Term Debt - Total
    dlc,        # Debt in Current Liabilities (Total Short-term)
    dd1,        # Long-Term Debt Due in 1 Year (The "Refinancing Wall")
    xint,       # Interest Expense - Total
    xinst,      # Interest Expense - Short-Term (Direct Fed Rates exposure)
    idit,       # Interest Income (Natural hedge against rate hikes)
    pstk,       # Preferred Stock (Quasi-debt, high duration risk)
    
    # --- 3. Balance Sheet: Assets (Liquidity & Stuffing) ---
    at,         # Total Assets
    act,        # Total Current Assets
    che,        # Cash & Short-Term Investments
    ivst,       # Short-Term Investments (Parking cash)
    rect,       # Receivables (Fraud signal: Channel Stuffing)
    invt,       # Inventories (Distress signal: Bloated stock)
    lifr,       # LIFO Reserve (Inflation interaction)
    intan,      # Intangibles (Soft assets)
    gdwl,       # Goodwill (Risk of massive write-down/Implosion)
    txdba,      # Deferred Tax Assets (Accounting aggression)
    
    # --- 4. Fixed Obligations & Macro Sensitivities ---
    lt,         # Total Liabilities
    lct,        # Total Current Liabilities
    mrct,       # Rental Commitments (5yr Total) - "Hidden" Operating Leverage
    fca,        # Foreign Exchange Income/Loss (Currency Macro interaction)
    capx,       # Capital Expenditures (Growth vs. Preservation)
    
    # --- 5. Equity & Payouts ---
    seq,        # Stockholders' Equity (Parent)
    mib,        # Noncontrolling Interest
    dv,         # Cash Dividends (Paper: ff_div_yld) [cite: 192]
    dvt,        # Dividends - Total (Includes preferred)
    xi,         # Extraordinary Items (Signal of "One-off" distress)
    ci,         # Comprehensive Income (Hidden losses)
    
    # --- 6. Market Data (For Market Cap & Returns) ---
    prcc_f,     # Price Close - Annual Fiscal
    csho,       # Common Shares Outstanding
    ajex        # Adjustment Factor (Cumulative) - MANDATORY
  ) |>
  collect()

# View the result: exact "public_date" tells you when the algorithm can "see" this data.
print(head(Data_Compustat_Annual_Raw))

##=================================##
## Local processing.
##=================================##

Data_Compustat_Annual_Final <- Data_Compustat_Annual_Raw |>
  mutate(
    datadate = ymd(datadate),
    
    # 1. Create a "Safe" Public Date for Annual Data
    # Statutory deadline is usually 90 days, but we use 4 months to be safe
    # and account for processing time/minor delays.
    safe_lag_date = datadate + months(4),
    
    # 2. Use 'repdte' if available (you must add it to your SELECT list first), 
    # otherwise default to the Safe Lag.
    # Note: If you didn't download 'repdte', just use 'safe_lag_date'.
    public_date = safe_lag_date 
  ) |>
  # 3. Ensure we don't use future data
  filter(public_date <= Sys.Date()) |>
  arrange(permno, datadate)

print(head(Data_Compustat_Final))

length(unique(Data_Compustat_Final$permno))

##=============================================================================#
##==== 4 - Output the Data ====================================================#
##=============================================================================#

##=================================##
## Save the compustat data.
##=================================##

Path <- file.path(Data_Compustat_Directory, "Data_Compustat_Annual_Raw.rds")
saveRDS(Data_Compustat_Annual_Raw, file = Path)

Path <- file.path(Data_Compustat_Directory, "Data_Compustat_Annual_Raw.rds")
saveRDS(Data_Compustat_Annual_Raw, file = Path)

##=============================================================================#
##=============================================================================#
##=============================================================================#