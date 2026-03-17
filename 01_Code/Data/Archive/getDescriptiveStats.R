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
              "RSQLite", "dbplyr", "kableExtra", "forcats",
              "patchwork"
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

## Directories for charts.
Charts_Directory <- file.path(Directory, "04_Charts/DescriptiveStatistics")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Date range.
start_date <- ymd("1965-01-01")
end_date <- ymd("2024-12-31")

## Charts.

width <- 3755
heigth <- 1833

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
## Load the delistings data.
##=================================##

Path <- file.path(Data_CRSP_Directory, "Data_Monthly_Complete.rds")
Data_Monthly_Complete <- readRDS(Path)

##=================================##
## Load the features (X variable space).
##=================================##

Path <- file.path(Data_Compustat_Directory, "Data_Compustat_Annual.rds")
Data_Compustat_Annual <- readRDS(Path)

##=================================##
## Load the annual returns per year.
##=================================##

Path <- file.path(Data_CRSP_Directory, "Data_y_annualized.rds")
Data_y_annualized <- readRDS(Path)

##=============================================================================#
##==== 3 - Descriptive Statistics =============================================#
##=============================================================================#

#==== 3A - Constituent Universe ===============================================#

tryCatch({
  
stats_overview <- tibble(
    Statistic = c(
      "Total Unique Firms (N)",
      "Total Observations",
      "Average Years per Firm",
      "Sample Start Date",
      "Sample End Date",
      "Total Identified Implosions"
    ),
    Value = c(
      format(n_distinct(Dataset$permno), big.mark = ","),
      format(nrow(Dataset), big.mark = ","),
      round(nrow(Dataset) / n_distinct(Data_y$permno), 0),
      as.character(min(Dataset$pred_date, na.rm = TRUE)),
      as.character(max(Dataset$pred_date, na.rm = TRUE)),
      format(sum(Dataset$y == 1), big.mark = ","))
  )
  
  # Output Overview Table
  print(
    kbl(stats_overview, format = "latex", booktabs = TRUE, 
        caption = "Aggregate Sample Statistics",
        align = c("l", "r")) |>
      kable_styling(latex_options = c("hold_position"))
  )
  
##================================##
## Sector distribution.
##================================##  


## Sectors.
desc_stats_naics <- stock_master_list_filtered |>
  mutate(
    NAICS_Sector = getSector(naics),
      Final_Sector = coalesce(NAICS_Sector),
    Class_System = if_else(!is.na(NAICS_Sector), "NAICS", "SIC (Proxy)")
  )

## Filter for all firms in the final dataset so we dont include any, which are not represented.
desc_stats_naics_filtered <- desc_stats_naics |>
  filter(permno %in% unique(Dataset$permno))

## Summarize the table.
table_naics <- desc_stats_naics_filtered |>
  group_by(Final_Sector) |>
  summarise(
    Count = n(),
    Percent = round(n() / nrow(desc_stats_naics_filtered) * 100, 1),
    System_Used = paste(unique(Class_System), collapse = "/")
  ) |>
  arrange(desc(Count))

# Output Table
kbl(table_naics, format = "latex", booktabs = TRUE, 
    caption = "Constituent Distribution by Industry Sector (NAICS/SIC Hybrid)",
    col.names = c("Industry Sector", "Count", "Percentage (%)", "Source")) |>
  kable_styling(latex_options = c("hold_position"))

}, error = function(e) message(e))

#==== 3B - Desciptive Stats of the Dataset ====================================#

tryCatch({
  
##===============================##
## Create a tibble with 1 obs for each firm.
##===============================##
  
CSI_Stats <- Dataset |>
    group_by(permno) |>
    summarise(csi_events_total = sum(y, na.rm = TRUE), .groups = "drop")

##===============================##
## Count how many times a firm enters a CSI event.
##===============================##

CSI_Distribution <- CSI_Stats |>
  count(csi_events_total, name = "number_of_firms")

CSI_Table_Data <- CSI_Distribution |>
  mutate(
    Percentage = number_of_firms / sum(number_of_firms) * 100
  )

kbl(CSI_Table_Data, 
    format = "latex", 
    booktabs = TRUE, 
    digits = 2, # Round percentages to 2 decimal places
    caption = "Distribution of Catastrophic Stock Implosion (CSI) Events per Firm",
    col.names = c("Total CSI Events", "Number of Firms", "Percentage (%)"),
    align = c("c", "r", "r")) |> # Center first col, right align numbers
  kable_styling(latex_options = c("hold_position", "striped"))

##===============================##
## Count the number of CSI events per year.
##===============================##

CSI_Yearly_Stats <- Dataset |>
  filter(y == 1) |>
  count(pred_year, name = "number_of_events") |>
  complete(pred_year = full_seq(pred_year, 1), fill = list(number_of_events = 0)) |>
  mutate(
    Percentage = number_of_events / sum(number_of_events) * 100
  )

kbl(CSI_Yearly_Stats, 
    format = "latex", 
    booktabs = TRUE, 
    digits = 2, 
    caption = "Temporal Distribution of Catastrophic Stock Implosions (Target Variable y)",
    col.names = c("Year", "Total Events", "Percentage (%)"),
    align = c("c", "r", "r")) |>
  kable_styling(latex_options = c("hold_position", "striped"))

##===============================##
## Count the number of CSI events per year (second table).
##===============================##

Full_Data <- CSI_Yearly_Stats |>
  select(pred_year, number_of_events, Percentage)

mid_point <- ceiling(nrow(Full_Data) / 2)

Left_Half <- Full_Data[1:mid_point, ]
Right_Half <- Full_Data[(mid_point + 1):nrow(Full_Data), ]

if (nrow(Right_Half) < nrow(Left_Half)) {
  Right_Half <- bind_rows(Right_Half, tibble(pred_year = NA, number_of_events = NA, Percentage = NA))
}

Split_Table <- bind_cols(Left_Half, Right_Half)

kbl(Split_Table, 
    format = "latex", 
    booktabs = TRUE, 
    digits = 2, 
    caption = "Temporal Distribution of CSI Active Years (Split View)",
    col.names = c("Year", "Events", "%", "Year", "Events", "%"),
    align = c("c", "r", "r", "c", "r", "r")) |>
  kable_styling(latex_options = c("hold_position", "striped")) |>
  add_header_above(c("Period 1" = 3, "Period 2" = 3))



##===============================##

}, error = function(e) message(e))

#==== 3C - Survival & Recovery Rate ===========================================#

tryCatch({
  
##===============================##
## Create a tibble with 1 obs for each firm.
##===============================##
  
  CSI_Stats <- Dataset |>
    group_by(permno) |>
    summarise(csi_events_total = sum(y, na.rm = TRUE), .groups = "drop")
  
  Last_Traded_Stats <- Data_Monthly_Complete |>
    group_by(permno) |>
    arrange(date) |>
    summarise(last_traded_year = year(last(date)), .groups = "drop")
  
  Performance_Stats <- Data_y_annualized |>
    group_by(permno) |>
    summarise(
      geometric_mean      = prod(1 + annual_ret, na.rm = TRUE)^(1 / n()) - 1,
      trading_years_total = n(),
      min_annual_ret      = min(annual_ret, na.rm = TRUE),
      max_annual_ret      = max(annual_ret, na.rm = TRUE),
      .groups = "drop"
    )
  
### The final tibble.
  Firm_Summary_Tibble <- Performance_Stats |>
    inner_join(Last_Traded_Stats, by = "permno") |>
    left_join(CSI_Stats, by = "permno") |>
    mutate(csi_events_total = replace_na(csi_events_total, 0)) |>
    select(permno, last_traded_year, geometric_mean, trading_years_total, csi_events_total, 
           min_annual_ret, max_annual_ret)
  
  glimpse(Firm_Summary_Tibble) 

##===============================##
## Classify the firms only by their geometric mean.
##===============================##
  Firm_Summary_Classified <- Firm_Summary_Tibble |>
    mutate(
      Growth_Category = case_when(
        geometric_mean > 0.10 ~ "High Growth (>10%)",
        geometric_mean > 0.05 & geometric_mean <= 0.10 ~ "Moderate Growth (5-10%)",
        geometric_mean >= 0.00 & geometric_mean <= 0.05 ~ "Low Growth (0-5%)",
        geometric_mean > -0.05 & geometric_mean < 0.00 ~ "Low Losses (0 to -5%)",
        geometric_mean <= -0.05 ~ "High Losses (<-5%)",
        TRUE ~ "Unknown"
      )
    )
  
### Split by whether a CSI event occured.
  Firm_Summary_Classified_split <- Firm_Summary_Classified |>
    mutate(Cohort = if_else(csi_events_total > 0, "Imploded Firms", "Never Imploded")) |>
    group_by(Cohort, Growth_Category) |>
    summarise(Count = n(), .groups = "drop") |>
    group_by(Cohort) |>
    mutate(
      Total_In_Cohort = sum(Count),
      Percentage = Count / Total_In_Cohort,
      Percentage_Fmt = scales::percent(Percentage, accuracy = 0.1)
    ) |>
    ungroup() |>
    select(Growth_Category, Cohort, Percentage_Fmt) |>
    tidyr::pivot_wider(names_from = Cohort, values_from = Percentage_Fmt, values_fill = "0.0%") |>
    mutate(sort_order = case_when(
      Growth_Category == "High Growth (>10%)" ~ 1,
      Growth_Category == "Moderate Growth (5-10%)" ~ 2,
      Growth_Category == "Low Growth (0-5%)" ~ 3,
      Growth_Category == "Low Losses (0 to -5%)" ~ 4,
      Growth_Category == "High Losses (<-5%)" ~ 5,
      TRUE ~ 6
    )) |>
    arrange(sort_order) |>
    select(-sort_order)
  
  Firm_Summary_Classified_split 
  
### Split by how many CSI events occured.
  Firm_Summary_By_Event_Count <- Firm_Summary_Classified |>
    group_by(csi_events_total, Growth_Category) |>
    summarise(Count = n(), .groups = "drop") |>
    group_by(csi_events_total) |>
    mutate(
      Total_In_Group = sum(Count),
      Percentage = Count / Total_In_Group,
      Percentage_Fmt = scales::percent(Percentage, accuracy = 0.1)
    ) |>
    ungroup() |>
    select(Growth_Category, csi_events_total, Percentage_Fmt) |>
    mutate(csi_events_total = paste0("Events: ", csi_events_total)) |>
    tidyr::pivot_wider(names_from = csi_events_total, values_from = Percentage_Fmt, values_fill = "0.0%") |>
    mutate(sort_order = case_when(
      Growth_Category == "High Growth (>10%)" ~ 1,
      Growth_Category == "Moderate Growth (5-10%)" ~ 2,
      Growth_Category == "Low Growth (0-5%)" ~ 3,
      Growth_Category == "Low Losses (0 to -5%)" ~ 4,
      Growth_Category == "High Losses (<-5%)" ~ 5,
      TRUE ~ 6
    )) |>
    arrange(sort_order) |>
    select(-sort_order)  
  
  Firm_Summary_By_Event_Count

##===============================##
## Classify the firms only by their geometric mean and their max/min growth rate.
##===============================##
  
  Firm_Summary_Classified_Additional <- Firm_Summary_Tibble |>
    mutate(
      Growth_Category = case_when(
        geometric_mean > 0.10 ~ "High Growth (>10%)",
        geometric_mean > 0.05 & geometric_mean <= 0.10 ~ "Moderate Growth (5-10%)",
        geometric_mean >= 0.02 & geometric_mean <= 0.05 ~ "Low Growth (2-5%)",
        geometric_mean <= 0.02 & geometric_mean > -0.02 & max_annual_ret < 0.15 ~ "Stagnation (-2-2%)",
        geometric_mean <= -0.02 & max_annual_ret < 0.02 ~ "Value Destruction (No Recovery)",
        geometric_mean <= -0.02 ~ "Value Destruction (<-2%)",
        
        TRUE ~ "Unknown"
      )
    )
  
  # --- Split and Summarize ---
  
  Firm_Summary_Classified_Additional_split <- Firm_Summary_Classified_Additional |>
    mutate(Cohort = if_else(csi_events_total > 0, "Imploded Firms", "Never Imploded")) |>
    group_by(Cohort, Growth_Category) |>
    summarise(Count = n(), .groups = "drop") |>
    group_by(Cohort) |>
    mutate(
      Total_In_Cohort = sum(Count),
      Percentage = Count / Total_In_Cohort,
      Percentage_Fmt = scales::percent(Percentage, accuracy = 0.1)
    ) |>
    ungroup() |>
    select(Growth_Category, Cohort, Percentage_Fmt) |>
    tidyr::pivot_wider(names_from = Cohort, values_from = Percentage_Fmt, values_fill = "0.0%") |>
    mutate(sort_order = case_when(
      Growth_Category == "High Growth (>10%)" ~ 1,
      Growth_Category == "Moderate Growth (5-10%)" ~ 2,
      Growth_Category == "Low Growth (2-5%)" ~ 3,
      Growth_Category == "Stagnation (-2-2%)" ~ 4,
      Growth_Category == "Value Destruction (<-2%)" ~ 5,
      Growth_Category == "Value Destruction (No Recovery)" ~ 6,
      TRUE ~ 7
    )) |>
    arrange(sort_order) |>
    select(-sort_order)
  
  Firm_Summary_Classified_Additional_split
  
### Now split by how many CSI events occured.
  Firm_Summary_By_Event_Count_additional <- Firm_Summary_Classified_Additional |>
    group_by(csi_events_total, Growth_Category) |>
    summarise(Count = n(), .groups = "drop") |>
    group_by(csi_events_total) |>
    mutate(
      Total_In_Group = sum(Count),
      Percentage = Count / Total_In_Group,
      Percentage_Fmt = scales::percent(Percentage, accuracy = 0.1)
    ) |>
    ungroup() |>
    select(Growth_Category, csi_events_total, Percentage_Fmt) |>
    mutate(csi_events_total = paste0("Events: ", csi_events_total)) |>
    tidyr::pivot_wider(names_from = csi_events_total, values_from = Percentage_Fmt, values_fill = "0.0%") |>
    mutate(sort_order = case_when(
      Growth_Category == "High Growth (>10%)" ~ 1,
      Growth_Category == "Moderate Growth (5-10%)" ~ 2,
      Growth_Category == "Low Growth (2-5%)" ~ 3,
      Growth_Category == "Stagnation (-2-2%)" ~ 4,
      Growth_Category == "Value Destruction (<-2%)" ~ 5,
      Growth_Category == "Value Destruction (No Recovery)" ~ 6,
      TRUE ~ 7
    )) |>
    arrange(sort_order) |>
    select(-sort_order) 

  Firm_Summary_By_Event_Count_additional  
  
  
  
##===============================##
## LaTeX Code.
##===============================## 
  
## First classification.
  
kbl(Firm_Summary_Classified_split, format = "latex", booktabs = TRUE, 
      caption = "Long-Term Growth Distribution: Imploded vs. Non-Imploded Firms",
      align = c("l", "r", "r")) |>
    kable_styling(latex_options = c("hold_position", "striped")) |>
    add_header_above(c(" " = 1, "Cohort Distribution" = 2))  
  
## First classification with columns via CSI events.
  
kbl(Firm_Summary_By_Event_Count, format = "latex", booktabs = TRUE, 
      caption = "Long-Term Growth Distribution by Frequency of Implosion Events",
      align = c("l", rep("r", ncol(Firm_Summary_By_Event_Count) - 1))) |>
    kable_styling(latex_options = c("hold_position", "striped", "scale_down")) |>
    add_header_above(c(" " = 1, "Number of CSI Events Experienced" = ncol(Firm_Summary_By_Event_Count) - 1))
  
## Second classification.
  
kbl(Firm_Summary_Classified_Additional_split, format = "latex", booktabs = TRUE, 
      caption = "Long-Term Growth Distribution: Imploded vs. Non-Imploded Firms",
      align = c("l", "r", "r")) |>
    kable_styling(latex_options = c("hold_position", "striped")) |>
    add_header_above(c(" " = 1, "Cohort Distribution" = 2))
  
## Second classification with columns via CSI events.

kbl(Firm_Summary_By_Event_Count_additional, format = "latex", booktabs = TRUE, 
    caption = "Long-Term Growth Distribution by Frequency of Implosion Events",
    align = c("l", rep("r", ncol(Firm_Summary_By_Event_Count_additional) - 1))) |>
  kable_styling(latex_options = c("hold_position", "striped", "scale_down")) |>
  add_header_above(c(" " = 1, "Number of CSI Events Experienced" = ncol(Firm_Summary_By_Event_Count_additional) - 1))


  
}, error = function(e) message(e))

##=============================================================================#
##==== 4 - Charts =============================================================#
##=============================================================================#

##==== 4A - Implosions over time ==============================================#

tryCatch({
  
implosions_by_year <- failed_companies |>
  mutate(year = year(implosion_date)) |>
  count(year, name = "count")

# 2. Prepare Data: Top 3 Industries per Year
failed_with_sector <- failed_companies |>
  left_join(desc_stats_naics |> select(permno, Final_Sector), by = "permno") |>
  mutate(year = year(implosion_date)) |>
  filter(!is.na(Final_Sector))

# Rank industries by count within each year and keep Top 3
top3_industries_per_year <- failed_with_sector |>
  group_by(year, Final_Sector) |>
  summarise(count = n(), .groups = "drop_last") |>
  mutate(rank = min_rank(desc(count))) |> # Rank 1 is highest count
  filter(rank <= 3) |>
  ungroup() |>
  arrange(year, desc(count))

p1 <- ggplot(implosions_by_year, aes(x = year, y = count)) +
  geom_col(fill = "#3474A5", width = 0.8) +
  # CHANGE 1: Set breaks to appear every 5 years
  scale_x_continuous(breaks = seq(min(implosions_by_year$year), 
                                  max(implosions_by_year$year), by = 5)) +
  labs(title = "",
       x = "Year", y = "Number of Implosions") +
  theme_bw() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    text = element_text(family = "serif", size = 12),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    
    # CHANGE 2: Make x-axis text smaller (e.g., size 9 or 10)
    axis.text.x = element_text(size = 10) 
  )

## Save the plot.
Path <- file.path(DescriptiveStats_Charts_Directory, "01_Defaults_over_Time.png")
ggsave(
  filename = Path,
  plot = p1,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

##===================================##
## Top Industries.
##===================================##

p2 <- ggplot(top3_industries_per_year, aes(x = year, y = count, fill = Final_Sector)) +
  geom_col(width = 0.8, color = "white", size = 0.2) + # White borders for stack separation
  scale_x_continuous(breaks = seq(min(top3_industries_per_year$year), 
                                  max(top3_industries_per_year$year), by = 5)) +
  # Use a distinct color palette (similar to the reference)
  scale_fill_manual(values = c(
    "Manufacturing" = "#C55A11", 
    "Services" = "#548235", 
    "Wholesale & Retail Trade" = "#2F5597",
    "Finance, Insurance & Real Estate" = "#7030A0",
    "Transportation & Utilities" = "#BF8F00",
    "Mining & Construction" = "#8FAADC",
    "Information" = "#A9D08E",
    "Other" = "gray"
    # Add more mapping pairs here based on your actual sector names
  )) +
  labs(title = "",
       x = "Year", y = "Number of Implosions",
       fill = "Industry") +
  theme_bw() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    text = element_text(family = "serif", size = 12),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 8)
  )

## Save the plot.
Path <- file.path(DescriptiveStats_Charts_Directory, "02_Top_Sectors.png")
ggsave(
  filename = Path,
  plot = p2,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

}, error = function(e) message(e))

##==== 4B - Implosion Cases per category ======================================#

tryCatch({
  
categories <- unique(Final_Analysis_Table$Outcome_Category)

# 2. Loop through each category
for (cat in categories) {
  
  cat(paste0("\nProcessing Category: ", cat, "..."))
  
  # --- A. Sampling ---
  cat_data <- Final_Analysis_Table |> filter(Outcome_Category == cat) |>
    filter(Cohort == "Imploded Firms")
  
  set.seed(123)
  if(nrow(cat_data) > 4) {
    sampled_events <- cat_data |> slice_sample(n = 4)
  } else {
    sampled_events <- cat_data
  }
  
  if(nrow(sampled_events) == 0) next
  
  # --- B. Define Zombie Rectangles (ROBUST MATCHING) ---
  
  # 1. Ensure type consistency
  sampled_events <- sampled_events |> mutate(permno = as.integer(permno))
  Data_Failed_Events <- Data_Failed_Events |> mutate(permno = as.integer(permno))
  
  zombie_rects <- sampled_events |>
    select(permno, start_year) |>
    # LEFT JOIN on Permno ONLY (One-to-Many expansion)
    left_join(Data_Failed_Events, by = "permno") |>
    
    # CALCULATE DISTANCE: 
    # Assume 'event_year' represents the end of that year (Dec 31).
    # We find the implosion date closest to that reference point.
    mutate(
      ref_date = make_date(start_year, 12, 31),
      date_diff = abs(as.numeric(implosion_date - ref_date))
    ) |>
    
    # GROUP & SLICE: Keep only the closest event for each sampled permno/year combo
    group_by(permno, start_year) |>
    slice_min(date_diff, n = 1, with_ties = FALSE) |>
    ungroup() |>
    
    # Construct Plotting Variables
    mutate(
      zombie_end = implosion_date %m+% months(18),
      label = paste0("Permno: ", permno, "\nEvent: ", start_year)
    ) 
  
  # Debug Check
  if(nrow(zombie_rects) == 0) {
    cat(" [Warning] No matching implosion events found (check permno overlap). Skipping.\n")
    next
  }
  
  # --- C. Prepare Price History Data ---
  
  target_permnos <- unique(zombie_rects$permno)
  
  plot_data <- Data_y |>
    mutate(permno = as.integer(permno)) |>
    filter(permno %in% target_permnos) |>
    select(permno, date, price) |>
    # Use inner_join to attach the label, ensuring we only keep relevant history
    inner_join(zombie_rects |> select(permno, label), 
               by = "permno", relationship = "many-to-many") |>
    arrange(permno, date)
  
  # --- D. Generate Grid Plot ---
  
  p_grid <- ggplot() +
    # 1. Shaded Region (Zombie Period)
    geom_rect(data = zombie_rects,
              aes(xmin = implosion_date, xmax = zombie_end, 
                  ymin = -Inf, ymax = Inf),
              fill = "firebrick", alpha = 0.2) +
    
    # 2. Price Line
    geom_line(data = plot_data, 
              aes(x = date, y = price), 
              color = "#2c3e50", linewidth = 0.5) +
    
    # 3. Facets
    facet_wrap(~label, scales = "free", ncol = 5) +
    
    # 4. Styling
    scale_y_continuous(labels = scales::dollar_format()) +
    labs(title = paste("Outcome:", cat),
         subtitle = "Red shaded area = 18-month post-implosion window",
         x = NULL, y = "Stock Price") +
    theme_bw() +
    theme(
      panel.grid.major = element_line(color = "grey92"),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(size = 6, angle = 45, hjust = 1),
      axis.text.y = element_text(size = 6),
      strip.background = element_rect(fill = "grey95"),
      strip.text = element_text(size = 7, face = "bold"),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10)
    )
  
  # --- E. Save Plot ---
  safe_filename <- str_replace_all(cat, "[^A-Za-z0-9]", "_") |>
    str_replace_all("__+", "_") 
  
  file_path <- file.path(Charts_Directory, paste0("03_Outcome_Sample_", safe_filename, ".png"))
  
  ggsave(
    filename = file_path,
    plot = p_grid,
    width = 3000, 
    height = 1800, 
    units = "px", 
    dpi = 300
  )
}

}, error = function(e) message(e))

##==== 4C - Dataset Imbalance =================================================#

tryCatch({
  
# A. Identify the total unique companies in the dataset
total_companies <- n_distinct(ml_panel$permno)

# B. Identify distinct Failure Events
# We look for unique combinations of Permno + Implosion Date.
# If a company fails in 1990 and again in 2000, this counts them twice (as requested).
# Note: Your current Step 5 uses `min(date)`, so currently only 1 failure per firm exists.
failure_events <- ml_panel |>
  filter(!is.na(implosion_date)) |>
  distinct(permno, implosion_date)

n_failures <- nrow(failure_events)

# C. Print the "True" Economic Imbalance
cat("--- Event-Level Imbalance ---\n")
cat("Total Unique Companies:  ", comma(total_companies), "\n")
cat("Total Failure Events:    ", comma(n_failures), "\n")
cat("Percent of Universe Died:", percent(n_failures / total_companies, accuracy = 0.01), "\n")


#=========================================================================#
# 2. Failure Rate Per Period (Correctly Normalized)
#=========================================================================#

# To get a "Rate per Year", we need:
# Numerator:   Number of companies that died in Year X
# Denominator: Number of companies that were "alive" (trading) in Year X

# A. Calculate the Denominator (Active Universe per Year)
active_counts <- ml_panel |>
  mutate(year = year(date)) |>
  group_by(year) |>
  summarise(n_active_firms = n_distinct(permno)) |>
  ungroup()

# B. Calculate the Numerator (Failures per Year)
# We use the 'failure_events' dataframe we created above
failure_counts <- failure_events |>
  mutate(year = year(implosion_date)) |>
  count(year, name = "n_failures")

# C. Merge and Calculate Rate
annual_stats <- active_counts |>
  left_join(failure_counts, by = "year") |>
  mutate(
    n_failures = replace_na(n_failures, 0), # Fill years with 0 failures
    failure_rate = n_failures / n_active_firms
  ) |>
  filter(year < 2024) # Optional: Cut off incomplete current year

#=========================================================================#
# 3. Visualization: The "Event" Dashboard
#=========================================================================#

# Color Palette
col_fail <- "#c0392b" # Red
col_safe <- "#2c3e50" # Blue

plot <- ggplot(annual_stats, aes(x = year)) +
  # Layer 1: The Active Universe (Background Bars)
  geom_col(aes(y = n_active_firms, fill = "Active Universe"), alpha = 0.3) +
  
  # Layer 2: The Failure Rate (Line Chart on Secondary Axis)
  # We scale the rate up by a factor (e.g., 20000) to make it visible on the same chart,
  # then reverse that scaling in the axis label.
  geom_line(aes(y = failure_rate * 20000, color = "Failure Rate"), size = 1.2) +
  geom_point(aes(y = failure_rate * 20000, color = "Failure Rate"), size = 2) +
  
  # Dual Axis Setup
  scale_y_continuous(
    name = "Number of Active Companies (Bars)",
    labels = comma,
    sec.axis = sec_axis(~ . / 20000, name = "Failure Rate (Red Line)", labels = percent)
  ) +
  
  scale_fill_manual(values = c("Active Universe" = col_safe)) +
  scale_color_manual(values = c("Failure Rate" = col_fail)) +
  
  labs(
    title = "",
    subtitle = "",
    x = "Year",
    fill = "", color = ""
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.title.y.right = element_text(color = col_fail, face = "bold"),
    axis.text.y.right = element_text(color = col_fail)
  )

## Save the plot.
Path <- file.path(DescriptiveStats_Charts_Directory, "04_Failure_Rate_Over_Time.png")
ggsave(
  filename = Path,
  plot = plot,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

}, error = function(e) message(e))

##=============================================================================#
##=============================================================================#
##=============================================================================#