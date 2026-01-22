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
## Load the features (X variable space).
##=================================##

Path <- file.path(Data_Compustat_Directory, "Data_Compustat_Annual.rds")
Data_Compustat_Annual <- readRDS(Path)

##=============================================================================#
##==== 3 - Descriptive Statistics =============================================#
##=============================================================================#

#==== 3A - Constituent Universe ===============================================#

## Generic.
stats_overview <- tibble(
  Statistic = c(
    "Total Unique Firms (N)",
    "Total Firm-Month Observations",
    "Average Months per Firm",
    "Sample Start Date",
    "Sample End Date",
    "Total Identified Implosions (Failures)",
    "Implosion Rate (Percentage of N)"
  ),
  Value = c(
    format(n_distinct(ml_panel$permno), big.mark = ","),
    format(nrow(ml_panel), big.mark = ","),
    round(nrow(ml_panel) / n_distinct(ml_panel$permno), 0),
    as.character(min(ml_panel$date)),
    as.character(max(ml_panel$date)),
    format(sum(ml_panel$TARGET_12M == 1, na.rm = TRUE) / 12, big.mark = ","), # Approx events (since target is 12m window)
    # OR better: use the count from your 'failed_companies' list
    format(nrow(failed_companies), big.mark = ",")
  )
)

# Add the Percentage Rate
stats_overview$Value[7] <- paste0(
  round((nrow(failed_companies) / n_distinct(ml_panel$permno)) * 100, 2), "%"
)

# 2. Generate LaTeX Table
kbl(stats_overview, format = "latex", booktabs = TRUE, 
    caption = "Aggregate Sample Statistics",
    align = c("l", "r")) |>
  kable_styling(latex_options = c("hold_position"))

## Sectors.
desc_stats_naics <- stock_master_list_filtered |>
  mutate(
    # Get the NAICS Name
    NAICS_Sector = getSector(naics),
      Final_Sector = coalesce(NAICS_Sector),
    
    # Flag which system was used (for transparency)
    Class_System = if_else(!is.na(NAICS_Sector), "NAICS", "SIC (Proxy)")
  )

table_naics <- desc_stats_naics |>
  group_by(Final_Sector) |>
  summarise(
    Count = n(),
    Percent = round(n() / nrow(desc_stats_naics) * 100, 1),
    System_Used = paste(unique(Class_System), collapse = "/")
  ) |>
  arrange(desc(Count))

# Output Table
kbl(table_naics, format = "latex", booktabs = TRUE, 
    caption = "Constituent Distribution by Industry Sector (NAICS/SIC Hybrid)",
    col.names = c("Industry Sector", "Count", "Percentage (%)", "Source")) |>
  kable_styling(latex_options = c("hold_position"))

#==== 3B - ML Dataset =========================================================#

##===================================##
## 1. Descriptive Stats: Safe vs Danger
##===================================##

# A. Identify numeric columns to analyze
# We exclude IDs and binary flags
numeric_cols <- Dataset |>
  select(where(is.numeric)) |>
  colnames()

vars_to_exclude <- c("permno", "y", "is_zombie_flag", "pred_year", "sich", "gvkey")
vars_to_analyze <- setdiff(numeric_cols, vars_to_exclude)

# B. Generate Statistics
desc_by_target <- Dataset |>
  # Create readable Group names based on 'y'
  mutate(Group = if_else(y == 1, "Danger (Pre-Crash)", "Safe/Normal")) |>
  select(Group, all_of(vars_to_analyze)) |>
  pivot_longer(cols = -Group, names_to = "Feature", values_to = "Value") |>
  group_by(Feature, Group) |>
  summarise(
    Mean = mean(Value, na.rm = TRUE),
    Median = median(Value, na.rm = TRUE),
    SD = sd(Value, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(Feature, Group)

# C. Format Table
final_desc_table <- desc_by_target |>
  pivot_wider(
    names_from = Group, 
    values_from = c(Mean, Median, SD),
    names_glue = "{Group}_{.value}"
  ) |>
  # Reorder columns for readability
  select(Feature, 
         contains("Safe/Normal_Mean"), contains("Danger (Pre-Crash)_Mean"),
         contains("Safe/Normal_SD"), contains("Danger (Pre-Crash)_SD"))

# D. Output
kbl(final_desc_table, format = "latex", booktabs = TRUE, digits = 3,
    caption = "Descriptive Statistics: Safe vs. Pre-Implosion Firms",
    col.names = c("Feature", "Mean (Safe)", "Mean (Danger)", "SD (Safe)", "SD (Danger)")) |>
  kable_styling(latex_options = c("scale_down", "hold_position"))

##===================================##
## 2. Survival and Recovery Rates
##===================================##

#### CODE IS WRONG.

# A. Identify Event Firms and define the "Target Year" (Event + 2)
Event_Firms <- Dataset |>
  filter(y == 1) |>
  select(permno, event_year = pred_year, mkt_cap_event = mkt_cap) |>
  mutate(target_year = event_year + 2)

# B. Find Outcome (Status 2 years later)
# We join specifically on (Permno) AND (Target Year == Pred Year)
Outcomes <- Event_Firms |>
  left_join(
    Dataset |> select(permno, pred_year, mkt_cap_future = mkt_cap),
    by = c("permno" = "permno", "target_year" = "pred_year")
  ) |>
  mutate(
    # If mkt_cap_future is NA, it means the firm was delisted/missing 2 years later
    is_survivor = !is.na(mkt_cap_future),
    
    # Calculate CAGR for survivors
    cagr_2yr = ifelse(is_survivor, (mkt_cap_future / mkt_cap_event)^(1/2) - 1, NA_real_)
  )

# C. Classify the Outcomes
Outcome_Classification <- Outcomes |>
  mutate(
    Outcome_Category = case_when(
      # Category 4: Dead (No data found 2 years later)
      !is_survivor ~ "Zombie State (Delisted/Inactive)",
      
      # Category 3: Implosion (Survived but lost value)
      is_survivor & cagr_2yr <= 0 ~ "Implosion (Stagnation)",
      
      # Category 2: Low Growth (Positive but <= 10%)
      is_survivor & cagr_2yr > 0 & cagr_2yr <= 0.10 ~ "Recovery (Low Growth)",
      
      # Category 1: Strong Growth (> 10%)
      is_survivor & cagr_2yr > 0.10 ~ "Recovery (Strong Growth)",
      
      TRUE ~ "Unknown"
    )
  )

# D. Generate Summary Table
outcome_table <- Outcome_Classification |>
  group_by(Outcome_Category) |>
  summarise(
    Count = n(),
    Percentage = n() / nrow(Outcome_Classification)
  ) |>
  mutate(Percentage = paste0(round(Percentage * 100, 2), "%")) |>
  arrange(desc(Count))

# Output
kbl(outcome_table, format = "latex", booktabs = TRUE, 
    caption = "Post-Implosion Outcomes (2-Year Horizon)",
    col.names = c("Outcome Category", "Count", "Percentage")) |>
  kable_styling(latex_options = "hold_position")

##===================================##
## 3. Frequency of Implosions
##===================================##

# A. Get Total Universe of Firms (All unique permnos in the cleaned set)
All_Firms <- unique(Dataset$permno)

# B. Count Events per Firm
Events_Per_Firm <- Dataset |>
  group_by(permno) |>
  summarise(total_events = sum(y, na.rm = TRUE)) |>
  ungroup()

# C. Categorize
implosion_freq_table <- Events_Per_Firm |>
  count(total_events) |>
  mutate(
    Category = case_when(
      total_events == 0 ~ "Never Imploded",
      total_events == 1 ~ "Single Implosion",
      total_events >= 2 ~ "Multiple Implosions"
    )
  ) |>
  group_by(Category) |>
  summarise(
    Count = sum(n),
    .groups = "drop"
  ) |>
  mutate(
    Percentage = Count / sum(Count) * 100,
    Category = factor(Category, levels = c("Never Imploded", "Single Implosion", "Multiple Implosions"))
  ) |>
  arrange(Category)

# D. Output
kbl(implosion_freq_table, format = "latex", booktabs = TRUE, digits = 2,
    caption = "Frequency of Catastrophic Implosions per Firm",
    col.names = c("Category", "Number of Firms", "Percentage (%)")) |>
  kable_styling(latex_options = "hold_position")

#==== 2C - Charts =============================================================#

##===================================##
## Implosions over time.
##===================================##

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

##===================================##
## Sample of implosions.
##===================================##

categories <- unique(Outcome_Classification$Outcome_Category)

# 2. Loop through each category
for (cat in categories) {
  
  # --- A. Sampling ---
  # Sample 10 Permnos from this specific category
  set.seed(123) # Ensure reproducibility
  
  target_permnos <- Outcome_Classification |>
    filter(Outcome_Category == cat) |>
    pull(permno)
  
  # Handle cases with fewer than 10 firms
  if(length(target_permnos) > 10) {
    sample_permnos <- sample(target_permnos, 10)
  } else {
    sample_permnos <- target_permnos
  }
  
  # Skip if no firms found (safety check)
  if(length(sample_permnos) == 0) next
  
  # --- B. Prepare Data ---
  # Filter the main daily/monthly data (Data_y) for these firms
  plot_data <- Data_y |>
    filter(permno %in% sample_permnos) |>
    # Select only necessary columns to keep it light
    select(permno, date, price, implosion_date) |>
    mutate(label = as.character(permno))
  
  # --- C. Define Zombie Rectangles ---
  # Define the 18-month shaded area starting from the Implosion Date
  zombie_rects <- plot_data |>
    group_by(permno) |>
    summarise(
      implosion_date = first(implosion_date),
      # End date = Implosion Date + 18 Months
      zombie_end = first(implosion_date) %m+% months(18),
      label = first(label)
    ) |>
    ungroup()
  
  # --- D. Generate Grid Plot ---
  p_grid <- ggplot(plot_data, aes(x = date, y = price)) +
    # 1. Shaded Region (Zombie Period)
    geom_rect(data = zombie_rects,
              aes(xmin = implosion_date, xmax = zombie_end, 
                  ymin = -Inf, ymax = Inf),
              fill = "#7b85f5", alpha = 0.5, inherit.aes = FALSE) +
    
    # 2. Price Line
    geom_line(color = "#1f77b4", linewidth = 0.6) +
    
    # 3. Facets (Grid Layout)
    facet_wrap(~label, scales = "free", ncol = 5) +
    
    # 4. Styling
    labs(title = paste("Outcome Category:", cat),
         subtitle = "Blue shaded area indicates the 18-month post-implosion window",
         x = NULL, y = NULL) +
    theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(size = 6, angle = 45, hjust = 1),
      axis.text.y = element_text(size = 6),
      strip.background = element_rect(fill = "white", color = "grey80"),
      strip.text = element_text(size = 8, face = "bold"),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10)
    )
  
  # --- E. Save Plot ---
  # Create a clean filename (replace spaces/slashes with underscores)
  safe_filename <- str_replace_all(cat, "[^A-Za-z0-9]", "_") |>
    str_replace_all("__+", "_") # Remove double underscores
  
  file_path <- file.path(Charts_Directory, paste0("03_Outcome_Sample_", safe_filename, ".png"))
  
  ggsave(
    filename = file_path,
    plot = p_grid,
    width = 3000, 
    height = 2000, 
    units = "px", 
    dpi = 300
  )
  
  print(paste("Saved plot:", file_path))
}

##===================================##
## Imbalance of the dataset.
##===================================##

#=========================================================================#
# 1. Event-Level Imbalance (The "Company Count")
#=========================================================================#

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
