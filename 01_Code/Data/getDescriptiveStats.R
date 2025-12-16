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
Data_Directory <- file.path(Path, "02_Data")
Data_CRSP_Directory <- file.path(Data_Directory, "CRSP")

Charts_Directory <- file.path(Path, "04_Charts")
DescriptiveStats_Charts_Directory <- file.path(Charts_Directory, "DescriptiveStatistics")

Functions_Directory <- file.path(Path, "01_Code/Data/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Date range.
start_date <- ymd("2023-01-01")
end_date <- ymd("2024-12-31")

## Charts.

width <- 3755
heigth <- 1833

##=============================================================================#
## 2. Descriptive Statistics
##=============================================================================#

#==== 2A - Constituent Universe ===============================================#

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

#==== 2A - ML Dataset =========================================================#

##===================================##
## Overall Implosion: 
##===================================##

numeric_cols <- ml_panel |>
  select(where(is.numeric)) |>
  colnames()

# Remove ID columns and non-predictive columns from the list
vars_to_analyze <- setdiff(numeric_cols, c("permno", "is_zombie", "months_dist", "TARGET_12M"))

# 2. Generate Summary Statistics by Group (Safe vs Danger)
# We exclude the 'is_zombie' rows (NA target) from this specific comparison
desc_by_target <- ml_panel |>
  filter(!is.na(TARGET_12M)) |>
  mutate(Group = if_else(TARGET_12M == 1, "Danger (Pre-Crash)", "Safe/Normal")) |>
  select(Group, all_of(vars_to_analyze)) |>
  pivot_longer(cols = -Group, names_to = "Feature", values_to = "Value") |>
  group_by(Feature, Group) |>
  summarise(
    Mean = mean(Value, na.rm = TRUE),
    Median = median(Value, na.rm = TRUE),
    SD = sd(Value, na.rm = TRUE),
    # P25 = quantile(Value, 0.25, na.rm = TRUE), # Optional
    # P75 = quantile(Value, 0.75, na.rm = TRUE), # Optional
    .groups = "drop"
  ) |>
  arrange(Feature, Group)

# 3. Format as a Comparison Table
# Pivot wider to put Safe vs Danger side-by-side
final_desc_table <- desc_by_target |>
  pivot_wider(
    names_from = Group, 
    values_from = c(Mean, Median, SD),
    names_glue = "{Group}_{.value}"
  ) |>
  select(Feature, 
         contains("Safe/Normal_Mean"), contains("Danger (Pre-Crash)_Mean"),
         contains("Safe/Normal_SD"), contains("Danger (Pre-Crash)_SD"))

# Output LaTeX Table
kbl(final_desc_table, format = "latex", booktabs = TRUE, digits = 3,
    caption = "Descriptive Statistics: Safe vs. Pre-Implosion Firms",
    col.names = c("Feature", "Mean (Safe)", "Mean (Danger)", "SD (Safe)", "SD (Danger)")) |>
  kable_styling(latex_options = c("scale_down", "hold_position"))

##===================================##
## Descriptive Stats: Survival and Recovery Rates. 
##===================================##

outcome_analysis <- failed_companies |>
  select(permno, implosion_date) |>
  # Join with the full panel to check for data availability after 18 months
  left_join(ml_panel |> 
              group_by(permno) |> 
              summarise(last_date = max(date), .groups = "drop"), 
            by = "permno") |>
  mutate(
    # Check if they survived 18 months (approx 548 days)
    months_survived = interval(implosion_date, last_date) %/% months(1),
    is_survivor = months_survived > 18
  )

# 2. Add Performance Data for Survivors
# We join the CAGR calculated in the previous 'survivor_performance' step.
# If a stock is NOT a survivor, CAGR will be NA.
outcome_analysis <- outcome_analysis |>
  left_join(survivor_performance |> select(permno, post_zombie_cagr), by = "permno")

# 3. Apply the 4-Category Classification Rules
outcome_classification <- outcome_analysis |>
  mutate(
    Outcome_Category = case_when(
      # Category 4: Zombie State (Implosion and Removal)
      # Did not survive the 18-month window (or delisted immediately after)
      !is_survivor ~ "Zombie State (Delisted)",
      
      # Category 3: Implosion (Zombie-state but no overall removal)
      # Survived, but negative returns (Value Trap)
      is_survivor & post_zombie_cagr <= 0 ~ "Implosion (Stagnation)",
      
      # Category 2: Recovery (Low returns)
      # Survived, positive growth but <= 10%
      is_survivor & post_zombie_cagr > 0 & post_zombie_cagr <= 0.10 ~ "Recovery (Low Growth)",
      
      # Category 1: Recovery (Strong Growth)
      # Survived, > 10% CAGR (The "Ecstasy" candidates)
      is_survivor & post_zombie_cagr > 0.10 ~ "Recovery (Strong Growth)",
      
      TRUE ~ "Unknown" # Safety catch
    )
  )

# 4. Generate the Summary Table
outcome_table <- outcome_classification |>
  group_by(Outcome_Category) |>
  summarise(
    Count = n(),
    Percentage = n() / nrow(outcome_classification)
  ) |>
  mutate(Percentage = paste0(round(Percentage * 100, 2), "%")) |>
  arrange(desc(Count)) # Sort by most common result

# 5. Output LaTeX Table
kbl(outcome_table, format = "latex", booktabs = TRUE, 
    caption = "Post-Implosion Outcomes: Survival and Recovery Rates",
    col.names = c("Outcome Category", "Count", "Percentage")) |>
  kable_styling(latex_options = c("hold_position"))

##===================================##
## Descriptive Stats: Multiple implosions.
##===================================##

implosion_counts <- monthly_signals |> # Use the object from Step 5
  arrange(permno, date) |>
  group_by(permno) |>
  mutate(
    # Check criteria again (Drawdown < -0.8)
    # We use a simplified check here just to count the "entries" into the crash zone
    is_crash = drawdown <= -0.8,
    
    # Lag the crash flag to find the "Starts"
    prev_crash = lag(is_crash, default = FALSE),
    
    # A "New Event" is when we are in a crash today, but weren't last month
    new_event = is_crash & !prev_crash
  ) |>
  summarise(
    total_implosions = sum(new_event, na.rm = TRUE)
  ) |>
  ungroup()

# 2. Generate Frequency Table
# How many firms crashed 0 times, 1 time, 2 times?
implosion_freq_table <- implosion_counts |>
  count(total_implosions) |>
  mutate(
    # Create the categories first
    Category = case_when(
      total_implosions == 0 ~ "Never Imploded",
      total_implosions == 1 ~ "Single Implosion",
      total_implosions >= 2 ~ "Multiple Implosions"
    )
  ) |>
  # CRITICAL STEP: Group by the new Category and Sum the counts
  group_by(Category) |>
  summarise(
    Count = sum(n),
    .groups = "drop"
  ) |>
  # Calculate percentages on the aggregated data
  mutate(
    Percentage = Count / sum(Count) * 100
  ) |>
  # Optional: Reorder levels so "Never" is first, "Multiple" is last
  mutate(Category = factor(Category, levels = c("Never Imploded", "Single Implosion", "Multiple Implosions"))) |>
  arrange(Category)

# 3. Output LaTeX
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

set.seed(123) # Set seed for reproducibility
sample_permnos <- sample(failed_companies$permno, 20)

# 2. Prepare the Time Series Data
plot_data <- crsp_monthly |>
  filter(permno %in% sample_permnos) |>
  inner_join(failed_companies |> select(permno, implosion_date), by = "permno") |>
  mutate(
    # Create a clean label for the facet titles (Permno)
    label = as.character(permno) 
  )

# 3. Prepare the "Zombie Zone" Rectangles
# This defines the 18-month window starting from the Implosion Date
zombie_rects <- plot_data |>
  group_by(permno) |>
  summarise(
    implosion_date = first(implosion_date),
    zombie_end = first(implosion_date) %m+% months(18),
    # Get y-axis limits for each plot to ensure the rectangle covers the full height
    max_price = max(prc, na.rm = TRUE), 
    label = as.character(first(permno))
  ) |>
  ungroup()

# 4. Generate the Grid Plot
p_grid <- ggplot(plot_data, aes(x = date, y = prc)) +
  # A. The Blue Shaded Region (Zombie Period)
  # We use -Inf and Inf for y to shade the entire vertical strip
  geom_rect(data = zombie_rects,
            aes(xmin = implosion_date, xmax = zombie_end, 
                ymin = -Inf, ymax = Inf),
            fill = "#7b85f5", alpha = 0.6, inherit.aes = FALSE) + # Matches the reference blue-purple
  
  # B. The Price Line
  geom_line(color = "#1f77b4", size = 0.6) + # Standard matplotlib blue
  
  # C. Faceting (The Grid)
  facet_wrap(~label, scales = "free", ncol = 5) + 
  
  # D. Formatting
  labs(title = "",
       x = NULL, y = NULL) + # Reference image has no axis labels
  theme_bw() +
  theme(
    # Clean up the grid to look like the reference
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 6, angle = 0),
    axis.text.y = element_text(size = 6),
    strip.background = element_rect(fill = "white", color = "grey80"), # Facet label box
    strip.text = element_text(size = 7),
    plot.title = element_text(hjust = 0.5, family = "serif", size = 14, face = "bold")
  )

## Save the plot.
Path <- file.path(DescriptiveStats_Charts_Directory, "03_Sample_of_Implosions.png")
ggsave(
  filename = Path,
  plot = p_grid,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

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
