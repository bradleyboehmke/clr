library(keras)
keras::k_clear_session()

dataset <- dataset_mnist()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

train_data <- array_reshape(train_data, c(60000, 28 * 28))
test_data <- array_reshape(test_data, c(10000, 28 * 28))
train_data <- train_data / 255
test_data <- test_data / 255
train_targets <- to_categorical(train_targets)
test_targets <- to_categorical(test_targets)

# define model
model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = 'relu', input_shape = ncol(train_data)) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "rmsprop",
  metrics = c("accuracy")
)

lr_finder <- function(object, x, y, min_lr = 1e-07, max_lr = 1e-01,
                      batch_size = 128, epochs = 5) {

  # create exponentially growing learning rates
  n_iter <- ceiling(epochs * (nrow(x) / batch_size))
  growth_constant <- 15
  lr_rates <- exp(seq(0, growth_constant, length.out = n_iter))
  lr_rates <- lr_rates / max(lr_rates)
  lr_rates <- lr_rates * max_lr

  # define custom callback class
  LogMetrics <- R6::R6Class("LogMetrics",
                            inherit = KerasCallback,
                            public = list(
                              loss = NULL,
                              acc = NULL,
                              on_batch_end = function(batch, logs=list()) {
                                self$loss <- c(self$loss, logs[["loss"]])
                              }
                            ))

  callback_lr_init <- function(logs){
    iter <<- 0
    lr_hist <<- c()
    iter_hist <<- c()
  }
  callback_lr_set <- function(batch, logs){
    iter <<- iter + 1
    LR <- lr_rates[iter] # if number of iterations > l_rate values, make LR constant to last value
    if(is.na(LR)) LR <- lr_rates[length(lr_rates)]
    k_set_value(model$optimizer$lr, LR)
  }
  callback_lr_log <- function(batch, logs){
    lr_hist <<- c(lr_hist, k_get_value(model$optimizer$lr))
    iter_hist <<- c(iter_hist, k_get_value(model$optimizer$iterations))
  }
  callback_lr <- callback_lambda(on_train_begin=callback_lr_init, on_batch_begin=callback_lr_set)
  callback_logger <- callback_lambda(on_batch_end=callback_lr_log)
  callback_log_acc_lr <- LogMetrics$new()

  object %>% fit(
    x, y,
    batch_size = batch_size, epochs = epochs, verbose = 0,
    callbacks = list(callback_lr, callback_logger, callback_log_acc_lr),
  )

  data.frame(
    iter = iter_hist,
    lr = lr_hist,
    loss = callback_log_acc_lr$loss
  )

}

clr_finder <- lr_finder(model, train_data, train_targets, batch_size = 32, epochs = 5)

# all points
clr_finder %>%
  ggplot(aes(lr, loss)) +
  geom_point() +
  scale_x_log10()

# moving average
clr_finder %>%
  dplyr::mutate(avg_loss = zoo::rollmean(loss, k = 10, fill = NA)) %>%
  ggplot(aes(lr, avg_loss)) +
  geom_line() +
  scale_x_log10()

# print the accumulated losses
dplyr::glimpse(history$train_history)
dplyr::glimpse(history$val_history)







library(keras)
dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)


model <- keras_model_sequential() %>%
  layer_dense(
    units = 64, activation = "relu",
    input_shape = dim(train_data)[[2]]
  ) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
   loss = "mse",
   metrics = c("mae")
 )

 callback_clr <- new_callback_cyclical_learning_rate(
   step_size = 32,
   base_lr = 0.099,
   max_lr = 0.1,
   gamma = 1,
   scale_fn = function(x) x + 0.1
 )

model %>% fit(
   train_data, train_targets,
   validation_data = list(test_data, test_targets),
   epochs = 10, verbose = 1,
   callbacks = list(callback_clr)
)
callback_clr$history
