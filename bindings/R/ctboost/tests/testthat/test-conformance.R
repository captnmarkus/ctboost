test_that("version-1 regression matches categorical and missing golden cases", {
  cases <- ctboost_cases("prepared_regression_v1.cases.json")
  model <- ctboost_load_predictor(ctboost_fixture(cases$artifact))
  rows <- ctboost_rows(cases$rows)

  expect_equal(model$num_features, 2L)
  expect_equal(
    ctboost_predict_raw(model, rows),
    as.numeric(unlist(cases$raw_predictions)),
    tolerance = 1e-12
  )
})

test_that("version-2 binary scores rates, probabilities, and labels", {
  cases <- ctboost_cases("prepared_binary_v2.cases.json")
  model <- ctboost_load_predictor(ctboost_fixture(cases$artifact))
  rows <- ctboost_rows(cases$rows)

  expect_equal(
    ctboost_predict_raw(model, rows),
    as.numeric(unlist(cases$raw_predictions)),
    tolerance = 1e-12
  )
  expect_equal(
    ctboost_predict_proba(model, rows),
    do.call(rbind, lapply(cases$probabilities, unlist)),
    tolerance = 1e-12
  )
  expect_equal(ctboost_predict_class(model, rows), unlist(cases$class_labels))
})

test_that("version-2 multiclass scores softmax and labels", {
  cases <- ctboost_cases("prepared_multiclass_v2.cases.json")
  model <- ctboost_load_predictor(ctboost_fixture(cases$artifact))
  rows <- ctboost_rows(cases$rows)

  expect_equal(
    ctboost_predict_raw(model, rows),
    do.call(rbind, lapply(cases$raw_predictions, unlist)),
    tolerance = 1e-12
  )
  expect_equal(
    ctboost_predict_proba(model, rows),
    do.call(rbind, lapply(cases$probabilities, unlist)),
    tolerance = 1e-12
  )
  expect_equal(ctboost_predict_class(model, rows), unlist(cases$class_labels))
})

test_that("raw feature-pipeline artifacts fail closed", {
  expect_error(
    ctboost_load_predictor(ctboost_fixture("raw_pipeline_v2.json")),
    "prepared numeric features only",
    fixed = TRUE
  )
})

test_that("duplicate keys are rejected", {
  expect_error(
    ctboost_load_predictor(ctboost_fixture("duplicate_prepared_flag_v2.json")),
    "duplicate object keys",
    fixed = TRUE
  )
})

test_that("loaded predictors cannot be mutated accidentally", {
  model <- ctboost_load_predictor(ctboost_fixture("prepared_binary_v2.json"))
  expect_error(model$num_features <- 0L)
})

test_that("cyclic trees are rejected before scoring", {
  source <- ctboost_fixture("prepared_binary_v2.json")
  document <- paste(readLines(source, warn = FALSE), collapse = "\n")
  invalid <- sub(
    '"left_child": 1', '"left_child": 0', document, fixed = TRUE
  )
  expect_false(identical(document, invalid))
  path <- tempfile(fileext = ".json")
  on.exit(unlink(path), add = TRUE)
  writeLines(invalid, path, useBytes = TRUE)
  expect_error(ctboost_load_predictor(path), "cycle or shared child", fixed = TRUE)
})

test_that("empty JSON objects are not accepted as empty arrays", {
  source <- ctboost_fixture("prepared_binary_v2.json")
  document <- paste(readLines(source, warn = FALSE), collapse = "\n")
  invalid <- sub(
    '"tree_learning_rates": [0.75, 0.25]',
    '"tree_learning_rates": {}',
    document,
    fixed = TRUE
  )
  expect_false(identical(document, invalid))
  path <- tempfile(fileext = ".json")
  on.exit(unlink(path), add = TRUE)
  writeLines(invalid, path, useBytes = TRUE)
  expect_error(ctboost_load_predictor(path), "must be an array", fixed = TRUE)
})

test_that("artifact size is enforced at the exact byte boundary", {
  path <- ctboost_fixture("prepared_binary_v2.json")
  size <- file.info(path)$size
  expect_s3_class(ctboost_load_predictor(path, size), "ctboost_json_predictor")
  expect_error(
    ctboost_load_predictor(path, size - 1),
    "size limit",
    fixed = TRUE
  )
})

test_that("bounded artifact text cannot be reinterpreted as a filename", {
  target <- normalizePath(
    ctboost_fixture("prepared_binary_v2.json"),
    winslash = "/",
    mustWork = TRUE
  )
  path <- tempfile(fileext = ".json")
  on.exit(unlink(path), add = TRUE)
  writeChar(target, path, eos = NULL, useBytes = TRUE)
  expect_error(
    ctboost_load_predictor(path, file.info(path)$size),
    "could not parse predictor JSON",
    fixed = TRUE
  )
})

test_that("heterogeneous and null class labels preserve JSON values", {
  source <- jsonlite::fromJSON(
    ctboost_fixture("prepared_binary_v2.json"), simplifyVector = FALSE
  )
  rows <- ctboost_rows(list(list(-1.0), list(1.0)))
  label_sets <- list(
    list(1, "positive"),
    list(NULL, "positive"),
    list(list(code = 1), "positive")
  )
  for (labels in label_sets) {
    source$class_labels <- labels
    path <- tempfile(fileext = ".json")
    on.exit(unlink(path), add = TRUE)
    jsonlite::write_json(source, path, auto_unbox = TRUE, null = "null")
    predictor <- ctboost_load_predictor(path)
    predicted <- ctboost_predict_class(predictor, rows)
    expect_type(predicted, "list")
    expect_length(predicted, 2L)
    expect_equal(predicted[[1L]], labels[[1L]])
    expect_equal(predicted[[2L]], labels[[2L]])
    repeated <- ctboost_predict_class(
      predictor,
      ctboost_rows(list(list(-1.0), list(-1.0)))
    )
    expect_type(repeated, "list")
    expect_length(repeated, 2L)
  }
})
