#' Plot cyclic learning rate callback information
#'
#' @param callback_clr An object of class `CyclicLR`.
#' @param granularity Either "epoch" or "iteration". We advise to use epoch
#'   as we find it easier to work with. The plot will look very similar (except
#'   for the x-axis scaling) for both options as long as you choosed
#'   \code{step_size} in \code{new_callback_cyclical_learning_rate()} to be more
#'   iterations than one epoch has.
#' @param backend Either "base" for base R or "ggplot2".
#' @param trans_y_axis Value passed to [ggplot2::scale_y_continuous()] as the
#'   `trans` argument. Only supported for `backend = "ggplot2"`.
#' @importFrom graphics plot
#' @examples
#' library(keras)
#' dataset <- dataset_boston_housing()
#' c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset
#'
#' mean <- apply(train_data, 2, mean)
#' std <- apply(train_data, 2, sd)
#' train_data <- scale(train_data, center = mean, scale = std)
#' test_data <- scale(test_data, center = mean, scale = std)
#'
#'
#' model <- keras_model_sequential() %>%
#'   layer_dense(
#'     units = 64, activation = "relu",
#'     input_shape = dim(train_data)[[2]]
#'   ) %>%
#'   layer_dense(units = 64, activation = "relu") %>%
#'   layer_dense(units = 1)
#' model %>% compile(
#'   optimizer = optimizer_rmsprop(lr = 0.001),
#'   loss = "mse",
#'   metrics = c("mae")
#' )
#'
#' callback_clr <- new_callback_cyclical_learning_rate(
#'   step_size = 32,
#'   base_lr = 0.001,
#'   max_lr = 0.006,
#'   gamma = 0.99,
#'   mode = "exp_range"
#' )
#' model %>% fit(
#'   train_data, train_targets,
#'   validation_data = list(test_data, test_targets),
#'   epochs = 10, verbose = 1,
#'   callbacks = list(callback_clr)
#' )
#' callback_clr$history
#' plot_clr_history(callback_clr, backend = "base")
#' @export
plot_clr_history <- function(callback_clr,
                             granularity = "epoch",
                             trans_y_axis = "identity") {
  checkmate::assert_class(callback_clr, "CyclicLR")
  checkmate::assert_choice(granularity, c("epoch", "iteration"))

  x <- rlang::sym(granularity)

}

#' @importFrom rlang !! .data
plot_clr_history_ggplot2 <- function(callback_clr,
                                     granularity = "epoch",
                                     trans_y_axis) {
  x <- rlang::sym(granularity)
  data <- dispatch_granularity(callback_clr, granularity)
  ggplot2::ggplot() +
    ggplot2::geom_line(ggplot2::aes(x = !!x, .data$max_lr, color = "max_lr"),
                       data = data
    ) +
    ggplot2::geom_line(ggplot2::aes(x = !!x, .data$lr, color = "lr"),
                       data = data
    ) +
    ggplot2::geom_line(ggplot2::aes(x = !!x, .data$base_lr, color = "base_lr"),
                       data = data
    ) +
    ggplot2::scale_color_manual(
      breaks = c("max_lr", "lr", "base_lr"),
      values = c("gray20", "green", "gray20"),
      name = ""
    ) +
    ggplot2::ylab("") +
    ggplot2::scale_y_continuous(trans = trans_y_axis)

}

#' Get the history depending on granularity
#' @param callback_clr A callback.
#' @param granularity Either "epoch" or "iteration".
#' @return
#' The history in a tabular format.
dispatch_granularity <- function(callback_clr, granularity) {
  callback_clr[[ifelse(granularity == "epoch", "history_epoch", "history")]]
}
