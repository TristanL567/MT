filterFunds <- function(Input){
  
  banned_patterns <- regex(
    "\\bTRUST\\b|\\bETF\\b|E\\s*T\\s*F|\\bFUND\\b|\\bNOTE\\b|\\bL\\.P\\.\\b", 
    ignore_case = TRUE
  )
  
  ## Remove.
  stock_master_cleaned <- Input |> 
    filter(
      !str_detect(issuernm, banned_patterns),
      # !str_detect(ticker, regex("ETF", ignore_case = TRUE)) 
    )
  
  ## Verification of removed ones.
  removed_stocks <- Input |>
    filter(str_detect(issuernm, banned_patterns)) |>
    select(permno, issuernm, securitytype)
  
  Output <- list(cleaned = stock_master_cleaned,
                 removed = removed_stocks)
  return(Output)
}