getCatastrophicImplosions <- function(Data_Monthly,
                                      param_C = -0.8, ## Determines the maximum drawdown, needs to be triggered first.
                                      param_M = -0.2, ## Further drawdown of 20% on
                                      param_T = 18,
                                      param_h = 12,
                                      ...){
  
### Compute the drawdowns.
Drawdowns <- Data_Monthly |>
    group_by(permno) |>
    arrange(date) |> 
    mutate(
      drawdown = as.vector(ComputeDrawdowns(
        matrix(replace_na(tot_ret_net_dvds, 0), ncol = 1), 
        geometric = TRUE
      ))
    ) |>
    ungroup()
  
### Apply the filtering conditions.
Drawdown_classified <- Drawdowns |>
  arrange(permno, date) |>
  group_by(permno) |>
  mutate(
    is_initial_crash = drawdown <= param_C,
    price_18m_later = lead(price, n = param_T),
    zombie_period_return = (price_18m_later / price) - 1,
    is_zombie_drop = zombie_period_return <= param_M,
    is_csi_event = is_initial_crash & is_zombie_drop,
    target_y = slide_lgl(
      is_csi_event,
      .f = any,
      .after = param_h,
      .complete = FALSE 
    ),
    y = as.integer(target_y)
  ) |>
  ungroup()

return(Drawdown_classified)
  
}