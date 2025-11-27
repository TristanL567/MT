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

packages <- c("here", "xts", "dplyr", "tidyr"
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


#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

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
