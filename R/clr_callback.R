#' Cyclical learning rate scheduler
#'
#' This callback implements a cyclical learning rate policy (CLR) where the
#' learning rate cycles between two boundaries with some constant frequency, as
#' detailed in \href{https://arxiv.org/abs/1506.01186}{Smith (2017)}. In
#' addition, supports scaled learning-rate bandwidths to automatically adjust
#' the learing rate boundaries upon plateau.
#'
#' @param base_lr Numeric indicating initial learning rate to apply as the lower
#' boundary in the cycle.
#'
#' @param max_lr Numeric indicating upper boundary in the cycle. Functionally,
#' it defines the cycle amplitude (\code{max_lr - base_lr}). The learning rate
#' at any cycle is the sum of \code{base_lr} and some scaling of the amplitude;
#' therefore \code{max_lr} may not actually be reached depending on scaling
#' function.
#'
#' @param step_size Integer inidicating the number of training iterations per
#' half cycle. Authors suggest setting step_size 2-8 x training iterations in
#' epoch.
#'
#' @param mode Character indicating one of the following options.. If `scale_fn`
#'   is not `NULL`, this argument is ignored.
#'   \itemize{
#'       \item{"triangular": }{A basic triangular cycle with no amplitude
#'       scaling.}
#'       \item{"triangular2": }{A basic triangular cycle that scales initial
#'       amplitude by half each cycle.}
#'       \item{"exp_range": }{A cycle that scales initial amplitude by
#'       \eqn{gamma^( cycle iterations)} at each cycle iteration.}
#'     }
#'
#' @param gamma Numeric indicating the constant to apply when
#' \code{mode = "exp_range"}. This scaling function applies
#' \eqn{gamma^(cycle iterations)}.
#'
#' @param scale_fn Custom scaling policy defined by a single argument anonymous
#'   function, where \code{0 <= scale_fn(x) <= 1} for all \code{x >= 0}. Mode
#'   paramater is ignored when applied. Default is \code{NULL}.
#'
#' @param scale_mode Character of "cycle" or "iterations". Defines whether
#' \code{scale_fn} is evaluated on cycle number or cycle iterations (training
#' iterations since start of cycle). Default is \code{"cycle"}.
#'
#' @param patience Integer indicating the number of epochs of training without
#' validation loss improvement that the callback will wait before it adjusts
#' \code{base_lr} and \code{max_lr}.
#'
#' @param factor Numeric vector of length one which will scale \code{max_lr} and
#' (if applicable according to \code{decrease_base_lr}) \code{base_lr} after
#' \code{patience} epochs without improvement in the validation loss.
#'
#' @param decrease_base_lr Boolean indicating whether \code{base_lr} should
#' also be scaled with \code{factor} or not. Default is \code{TRUE}.
#'
#' @param cooldown Number of epochs to wait before resuming normal operation
#' after learning rate has been reduced.
#'
#' @details
#' This callback is general in nature and allows for:
#'
#' \itemize{
#'   \item{Constant learning rates: }{Allows you the same control as supplying a
#'   constant learning rate.}
#'   \item{Cyclical learning rates: }{Currently, the available cycical learning
#'   rates provided include:
#'     \itemize{
#'       \item{"triangular": }{A basic triangular cycle with no amplitude
#'       scaling.}
#'       \item{"triangular2": }{A basic triangular cycle that scales initial
#'       amplitude by half each cycle.}
#'       \item{"exp_range": }{A cycle that scales initial amplitude by
#'       \eqn{gamma^(cycle iterations)} at each cycle iteration.}
#'     }
#'     }
#'   \item{Decaying learning rates: }{Depending on validation loss such as
#'   \link{keras::callback_reduce_lr_on_plateau()}.}
#'   \item{Learning rates with scaled bandwidths. }{The arguments
#'   \code{patience}, \code{factor} and \code{decrease_base_lr} allow the user
#'   control over if and when the boundaries of the learning rate are adjusted.
#'   This feature allows you to combine decaying learning rates with cyclical
#'   learning rates. Typically, one wants to reduce the learning rate bandwith
#'   after validation loss has stopped improving for some time. Note that both
#'   \code{factor < 1} and \code{patience < Inf} must hold in order for this
#'   feature to take effect.
#' }
#'
#' For more details, please see \href{https://arxiv.org/abs/1506.01186}{Smith
#' (2017)}.
#'
#' @return
#' The callback object is a mutable R6 class of CyclicLR. This object will
#' return two main data frames of interest:
#'
#' \describe{
#'   \item{\code{history} data frame}{Contains loss and metric information
#'   along with the actual learning rate value for each iteration.}
#'   \item{\code{history_epoch} data frame}{Contains loss and metric information
#'   along with learning rate meta data for each epoch.}
#' }
#'
#' @family callbacks
#' @references
#' Smith, L.N. Cycical Learning Rates for Training Neural Networks. arXiv
#' preprint arXiv:1506.01186 (2017). https://arxiv.org/abs/1506.01186
#'
#' Lorenz Walthert (2020). KerasMisc: Add-ons for Keras. R package
#' version 0.0.0.9001. https://github.com/lorenzwalthert/KerasMisc
#' @export
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
new_callback_cyclical_learning_rate <- function(
  base_lr = 0.001,
  max_lr = 0.006,
  step_size = 2000,
  mode = "triangular",
  gamma = 1,
  scale_fn = NULL,
  scale_mode = "cycle",
  patience = Inf,
  factor = 0.9,
  decrease_base_lr = TRUE,
  cooldown = 2
) {
  CyclicLR$new(
    base_lr = base_lr,
    max_lr = max_lr,
    step_size = step_size,
    mode = mode,
    gamma = gamma,
    scale_fn = scale_fn,
    scale_mode = scale_mode,
    patience = patience,
    factor = factor,
    decrease_base_lr = decrease_base_lr,
    cooldown = cooldown
  )
}

#' @importFrom utils write.table
CyclicLR <- R6::R6Class("CyclicLR",
                        inherit = keras::KerasCallback,
                        public = list(
                          base_lr = NULL,
                          max_lr = NULL,
                          step_size = NULL,
                          mode = NULL,
                          gamma = NULL,
                          scale_fn = NULL,
                          scale_mode = NULL,
                          patience = NULL,
                          factor = NULL,
                          clr_iteration = NULL,
                          trn_iteration = NULL,
                          history = NULL,
                          history_epoch = NULL,
                          trn_epochs = NULL,
                          not_improved_for_n_times = NULL,
                          decrease_base_lr = NULL,
                          cooldown = NULL,
                          cooldown_counter = NULL,
                          initialize = function(base_lr = 0.001,
                                                max_lr = 0.006,
                                                step_size = 2000,
                                                mode = "triangular",
                                                gamma = 1,
                                                scale_fn = NULL,
                                                scale_mode = "cycle",
                                                patience = 3,
                                                factor = 0.9,
                                                decrease_base_lr = TRUE,
                                                cooldown = 0) {

                            assert_CyclicLR_init(
                              base_lr,
                              max_lr,
                              step_size,
                              mode,
                              gamma,
                              scale_fn,
                              scale_mode,
                              patience,
                              factor,
                              decrease_base_lr,
                              cooldown
                            )

                            self$base_lr <- base_lr
                            self$max_lr <- max_lr
                            self$step_size <- step_size
                            self$mode <- mode
                            self$gamma <- gamma

                            if (is.null(scale_fn)) {
                              if (self$mode == "triangular") {
                                self$scale_fn <- function(x) 1
                                self$scale_mode <- "cycle"
                              } else if (self$mode == "triangular2") {
                                self$scale_fn <- function(x) 1 / (2^(x - 1))
                                self$scale_mode <- "cycle"
                              } else if (self$mode == "exp_range") {
                                self$scale_fn <- function(x) gamma^(x)
                                self$scale_mode <- "iteration"
                              }
                            } else {
                              self$scale_fn <- scale_fn
                              self$scale_mode <- scale_mode
                            }
                            self$clr_iteration <- 0
                            self$trn_iteration <- 0
                            self$trn_epochs <- 1
                            self$history <- data.frame()
                            self$history_epoch <- data.frame()
                            self$patience <- patience
                            self$factor <- factor
                            self$decrease_base_lr <- decrease_base_lr
                            self$cooldown <- cooldown
                            self$cooldown_counter <- 0
                            self$.reset()
                          },

                          .reset = function(new_base_lr = NULL,
                                            new_max_lr = NULL,
                                            new_step_size = NULL) {

                            if (!is.null(new_base_lr)) {
                              self$base_lr <- new_base_lr
                            }
                            if (!is.null(new_max_lr)) {
                              self$max_lr <- new_max_lr
                            }
                            if (!is.null(new_step_size)) {
                              self$step_size <- new_step_size
                            }
                            self$clr_iteration <- 0
                            self$not_improved_for_n_times <- 0
                          },
                          clr = function() {
                            cycle <- floor(1 + self$clr_iteration / (2 * self$step_size))
                            x <- abs(self$clr_iteration / self$step_size - 2 * cycle + 1)
                            if (self$scale_mode == "cycle") {
                              self$base_lr +
                                (self$max_lr - self$base_lr) *
                                max(0, (1 - x)) *
                                self$scale_fn(cycle)

                            } else {
                              self$base_lr +
                                (self$max_lr - self$base_lr) * max(0, (1 - x)) * self$scale_fn(self$clr_iteration)
                            }
                          },
                          in_cooldown = function() {
                            self$cooldown_counter > 0
                          },

                          on_train_begin = function(logs) {
                            if (self$clr_iteration == 0) {
                              k_set_value(self$model$optimizer$lr, self$base_lr)
                            } else {
                              k_set_value(self$model$optimizer$lr, self$clr())
                            }
                          },

                          on_batch_end = function(batch, logs = list()) {
                            new_history <- as.data.frame(do.call(cbind, c(
                              lr = k_get_value(self$model$optimizer$lr),
                              base_lr = self$base_lr,
                              max_lr = self$max_lr,
                              iteration = self$trn_iteration,
                              epochs = self$trn_epochs,
                              logs
                            )))

                            cols <- names(new_history)
                            relevant <- cols[!(cols %in% c("batch", "size"))]
                            new_history <- subset(new_history, select = relevant)

                            self$history <- rbind(
                              self$history, new_history
                            )
                            self$trn_iteration <- self$trn_iteration + 1
                            self$clr_iteration <- self$clr_iteration + 1
                            k_set_value(self$model$optimizer$lr, self$clr())
                          },
                          on_epoch_end = function(epochs, logs = list()) {
                            self$trn_epochs = self$trn_epochs + 1
                            best <- ifelse(nrow(self$history_epoch) > 0,
                                           min(self$history_epoch$val_loss),
                                           logs$val_loss
                            )
                            if (logs$val_loss > best) {
                              self$not_improved_for_n_times <- self$not_improved_for_n_times + 1
                            } else {
                              self$not_improved_for_n_times <- 0
                            }
                            self$history_epoch <- rbind(
                              self$history_epoch,
                              as.data.frame(do.call(cbind, c(
                                logs, epoch = epochs,
                                not_improved_for_n_times = self$not_improved_for_n_times,
                                base_lr = self$base_lr, max_lr = self$max_lr,
                                lr = k_get_value(self$model$optimizer$lr),
                                in_cooldown = self$in_cooldown()
                              )))
                            )
                            if (self$in_cooldown()) {
                              self$cooldown_counter <- self$cooldown_counter - 1L
                            } else if (self$not_improved_for_n_times > self$patience) {
                              if (self$decrease_base_lr) {
                                self$base_lr <-  self$factor * self$base_lr
                                self$cooldown_counter <- self$cooldown
                              }
                              candidate_max_lr <- self$factor * self$max_lr
                              if (self$base_lr <= candidate_max_lr) {
                                self$max_lr <- candidate_max_lr
                              }
                            }
                          }
                        )
)


assert_CyclicLR_init <- function(
  base_lr,
  max_lr,
  step_size,
  mode,
  gamma,
  scale_fn,
  scale_mode,
  patience,
  factor,
  decrease_base_lr,
  cooldown
) {

  checkmate::assert_number(max_lr - base_lr, lower = 0)
  checkmate::assert_number(step_size, lower = 1)
  checkmate::assert_number(gamma)
  checkmate::assert_number(patience)
  checkmate::assert_number(factor)
  checkmate::assert_logical(decrease_base_lr)
  checkmate::assert_number(cooldown, lower = 0)
  if (is.null(scale_fn)) {
    checkmate::assert_choice(
      mode,
      choices = c("triangular", "triangular2", "exp_range")
    )
  } else {
    checkmate::assert_function(scale_fn)
  }
}
