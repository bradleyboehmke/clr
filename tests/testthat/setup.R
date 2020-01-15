library(keras)
keras::k_clear_session()

dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

generate_model <- function() {
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
}
