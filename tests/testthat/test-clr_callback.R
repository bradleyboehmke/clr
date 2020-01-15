test_that("clr_callback assertions", {
  expect_error(
    new_callback_cyclical_learning_rate(
      step_size = -1,
      base_lr = 0.001,
      max_lr = 0.006,
      gamma = 0.99,
      mode = "exp_range"
    ), ">= 1"
  )

  expect_error(new_callback_cyclical_learning_rate(
    step_size = 6,
    base_lr = 0.01,
    max_lr = 0.006,
    gamma = 0.99,
    mode = "exp_range"
  ), ">= 0")

  expect_error(new_callback_cyclical_learning_rate(
    step_size = 6,
    base_lr = 0.001,
    max_lr = 0.006,
    gamma = 0.99,
    mode = "abc"
  ), "exp_range")
})

test_that("triangle", {
  base_lr <- 0.001
  max_lr <- 0.006
  callback_clr <- new_callback_cyclical_learning_rate(
    step_size = 6,
    base_lr = base_lr,
    max_lr = max_lr
  )
  generate_model() %>% fit(
    train_data, train_targets,
    validation_data = list(test_data, test_targets),
    epochs = 10, verbose = 0,
    callbacks = list(callback_clr)
  )

  expect_gte(
    min(callback_clr$history$lr), base_lr
  )

})
