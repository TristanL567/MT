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
              "RSQLite", "dbplyr", "lubridate", "openxlsx"
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
Data_VariableDefinitions_Directory <- file.path(Directory, "01_Code/Data/Variables_Annual.xlsx")

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

##=================================##
## Variables definitions (stored in Excel).
##=================================##

balancesheet_variables <- read.xlsx(Data_VariableDefinitions_Directory, sheet = "balancesheet_annual")
incomestatement_variables <- read.xlsx(Data_VariableDefinitions_Directory, sheet = "incomestatement_annual")
other_variables <- read.xlsx(Data_VariableDefinitions_Directory, sheet = "other_annual")

##=============================================================================#
##==== 3 - Compustat Data =====================================================#
##=============================================================================#

##==== 3A - Preparation for the Data Download =================================#

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
    lpermno %in% id, 
    linktype %in% c("LC", "LU", "LS"), 
    linkprim %in% c("P", "C")
  ) |>
  select(gvkey, permno = lpermno, linkdt, linkenddt)

##==== 3B - Balance Sheet Data ================================================#

balancesheet_identifier <- balancesheet_variables$identifier

##=================================##
## Fetch the balance-sheet data.
##=================================##

Data_Compustat_Annual_BalanceSheet <- comp_a_db |>
  inner_join(valid_links, by = "gvkey") |> 
  filter(
    indfmt == "INDL", datafmt == "STD", consol == "C",
    datadate >= start_date, 
    datadate >= linkdt & (is.na(linkenddt) | datadate <= linkenddt)
  ) |>
  select(
    # --- Identifiers & Meta ---
    permno, gvkey, datadate, fyear, sich,
    
    # --- Balance Sheet: Assets ---
    at,       # Assets - Total
    act,      # Current Assets - Total
    che,      # Cash and Short-Term Investments
    ivst,     # Short-Term Investments - Total
    rect,     # Receivables - Total
    invt,     # Inventories - Total
    aco,      # Current Assets - Other - Total
    
    # --- Non-Current Assets ---
    ppent,    # Property, Plant and Equipment - Total (Net)
    ivpt,     # Investments - Permanent - Total
    intan,    # Intangible Assets - Total
    gdwl,     # Goodwill
    txdba,    # Deferred Tax Asset - Long Term
    ao,       # Assets - Other
    
    # --- Balance Sheet: Liabilities ---
    lt,       # Liabilities - Total
    lct,      # Current Liabilities - Total
    ap,       # Accounts Payable - Trade
    txp,      # Income Taxes Payable
    drc,      # Deferred Revenue - Current
    dlc,      # Debt in Current Liabilities - Total
    dd1,      # Long-Term Debt Due in One Year
    
    # --- Non-Current Liabilities ---
    dltt,     # Long-Term Debt - Total
    txditc,   # Deferred Taxes and Investment Tax Credit
    lo,       # Liabilities - Other - Total
    mib,      # Noncontrolling Interest (Balance Sheet)
    
    # --- Balance Sheet: Equity ---
    seq,      # Stockholders Equity - Parent
    cstk,     # Common/Ordinary Stock (Capital)
    caps,     # Capital Surplus/Share Premium
    pstk,     # Preferred/Preference Stock (Capital) - Total
    re,       # Retained Earnings
    acominc,  # Accumulated Other Comprehensive Income (Loss)
    tstk      # Treasury Stock - Total
  ) |>
  collect()

##=================================##
## Save the data.
##=================================##

Path <- file.path(Data_Compustat_Directory, "Data_Compustat_Annual_BalanceSheet.rds")
saveRDS(Data_Compustat_Annual_BalanceSheet, file = Path)

##==== 3C - Income Statement Data =============================================#

incomestatement_identifier <- incomestatement_variables$identifier

##=================================##
## Fetch the balance-sheet data.
##=================================##

Data_Compustat_Annual_IncomeStatement <- comp_a_db |>
  inner_join(valid_links, by = "gvkey") |> 
  filter(
    indfmt == "INDL", datafmt == "STD", consol == "C",
    datadate >= start_date, 
    datadate >= linkdt & (is.na(linkenddt) | datadate <= linkenddt)
  ) |>
  select(
    # --- Identifiers & Meta ---
    permno, gvkey, datadate, fyear, sich,
    
    # --- Top Line & Margins ---
    sale,     # Net Sales
    cogs,     # Cost of Goods Sold
    gp,       # Gross Profit
    
    # --- Operating Expenses ---
    xsga,     # Selling, General and Administrative Expenses
    xrd,      # Research and Development Expense
    dp,       # Depreciation and Amortization
    spi,      # Special Items
    
    # --- Operating Profit ---
    ebit,     # Earnings Before Interest & Taxes
    
    # --- Non-Operating / Interest ---
    xint,     # Interest and Related Expense - Total
    xinst,    # Interest Expense - Short-Term
    idit,     # Interest Income
    nopi,     # Non-Operating Income (Expense)
    fca,      # Foreign Exchange Income/Loss
    
    # --- Pretax & Taxes ---
    pi,       # Pretax Income
    txt,      # Income Taxes - Total
    
    # --- Bottom Line / EPS ---
    xi,       # Extraordinary Items
    ni,       # Net Income (Loss)
    epsfi,    # Earnings Per Share (Basic) - Excl. Extra Items
    
    # --- Comprehensive / Payouts ---
    ci,       # Comprehensive Income - Total
    dv,       # Cash Dividends (FIXED: Added back)
    dvt       # Dividends - Total
  ) |>
  collect()

##=================================##
## Save the data.
##=================================##

Path <- file.path(Data_Compustat_Directory, "Data_Compustat_Annual_IncomeStatement.rds")
saveRDS(Data_Compustat_Annual_IncomeStatement, file = Path)

##==== 3D - Other Data ========================================================#

other_identifier <- other_variables$identifier

##=================================##
## Fetch the balance-sheet data.
##=================================##

Data_Compustat_Annual_Other <- comp_a_db |>
  inner_join(valid_links, by = "gvkey") |> 
  filter(
    indfmt == "INDL", datafmt == "STD", consol == "C",
    datadate >= start_date, 
    datadate >= linkdt & (is.na(linkenddt) | datadate <= linkenddt)
  ) |>
  select(
    # --- Identifiers & Meta ---
    permno, gvkey, datadate, fyear, sich,
    
    # --- Real Economy / Efficiency ---
    emp,      # Employees
    
    # --- Profitability Adjustments ---
    oibdp,    # Operating Income Before Depreciation (EBITDA)

    # --- Discretionary Expenses ---
    xad,      # Advertising Expense
    xrent,    # Rental Expense
    
    # --- Asset Age & Quality ---
    ppegt,    # Property, Plant and Equipment - Total (Gross)
    
    # --- Liquidity & Valuation ---
    wcap,     # Working Capital (Balance Sheet)
    mkvalt,   # Market Value - Total (Fiscal)
    prcc_f,   # Price Close - Annual Fiscal (ADDED)
    csho,     # Common Shares Outstanding (ADDED)
    ajex      # Adjustment Factor (Company) - Cumulative by Ex-Date
  ) |>
  collect()

##=================================##
## Save the data.
##=================================##

Path <- file.path(Data_Compustat_Directory, "Data_Compustat_Annual_Other.rds")
saveRDS(Data_Compustat_Annual_Other, file = Path)

##=============================================================================#
##==== 4 - Output the Consolidated Data =======================================#
##=============================================================================#

##=================================##
## Merge the datasets.
##=================================##

join_keys <- c("permno", "gvkey", "datadate", "fyear", "sich")

Data_Compustat_Annual <- Data_Compustat_Annual_BalanceSheet |>
  inner_join(
    Data_Compustat_Annual_IncomeStatement, 
    by = join_keys
  ) |>
  inner_join(
    Data_Compustat_Annual_Other, 
    by = join_keys
  ) |>
  arrange(permno, datadate) |>
  distinct()

# View the structure of the final master dataset
glimpse(Data_Compustat_Annual)

##=================================##
## Save the consolidated data.
##=================================##

Path <- file.path(Data_Compustat_Directory, "Data_Compustat_Annual.rds")
saveRDS(Data_Compustat_Annual, file = Path)

##=============================================================================#
##=============================================================================#
##=============================================================================#