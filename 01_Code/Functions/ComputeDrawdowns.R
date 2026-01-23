#' Calculates drawdowns for a numeric matrix of returns.
#'
#' Return Drawdowns
#'
#' This function computes the drawdowns of a matrix of asset returns.
#' @param R A numeric matrix where rows represent time periods and columns represent assets.
#' @param geometric A boolean indicating whether to use geometric (TRUE) or arithmetic (FALSE) compounding.
#' @export

ComputeDrawdowns <- function(R, geometric = TRUE) {

  if (!is.matrix(R) || !is.numeric(R)) {
    stop("Input 'R' must be a numeric matrix.")
  }

  drawdown_matrix <- apply(R, 2, function(x_col) {
        if (geometric) {
      Return.cumulative <- cumprod(1 + x_col)
    } else {
      Return.cumulative <- 1 + cumsum(x_col)
    }

    maxCumulativeReturn <- cummax(c(1, Return.cumulative))[-1]
        drawdown <- Return.cumulative / maxCumulativeReturn - 1
    return(drawdown)
  })
    return(drawdown_matrix)
}
