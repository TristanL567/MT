getSector <- function(n_code) {
    prefix <- substr(as.character(n_code), 1, 2)
    
    case_when(
      prefix == "11" ~ "Agriculture, Forestry, Fishing & Hunting",
      prefix == "21" ~ "Mining, Quarrying, and Oil & Gas Extraction",
      prefix == "22" ~ "Utilities",
      prefix == "23" ~ "Construction",
      prefix %in% c("31","32","33") ~ "Manufacturing",
      prefix == "42" ~ "Wholesale Trade",
      prefix %in% c("44","45") ~ "Retail Trade",
      prefix %in% c("48","49") ~ "Transportation and Warehousing",
      prefix == "51" ~ "Information",
      prefix == "52" ~ "Finance and Insurance",
      prefix == "53" ~ "Real Estate and Rental and Leasing",
      prefix == "54" ~ "Professional, Scientific, & Technical Services",
      prefix == "55" ~ "Management of Companies and Enterprises",
      prefix == "56" ~ "Admin. Support & Waste Mgmt",
      prefix == "61" ~ "Educational Services",
      prefix == "62" ~ "Health Care and Social Assistance",
      prefix == "71" ~ "Arts, Entertainment, and Recreation",
      prefix == "72" ~ "Accommodation and Food Services",
      prefix == "81" ~ "Other Services (except Public Admin)",
      prefix == "92" ~ "Public Administration",
      TRUE ~ NA_character_ # Return NA if no match or missing
    )
}