.ctb_stop <- function(message) {
  stop(message, call. = FALSE)
}

.ctb_is_object <- function(value) {
  is.list(value) && !is.null(names(value))
}

.ctb_is_array <- function(value) {
  is.list(value) && is.null(names(value))
}

.ctb_reject_duplicate_keys <- function(value) {
  if (!is.list(value)) {
    return(invisible(NULL))
  }
  if (!is.null(names(value)) && anyDuplicated(names(value)) != 0L) {
    .ctb_stop("predictor JSON contains duplicate object keys")
  }
  for (child in value) {
    .ctb_reject_duplicate_keys(child)
  }
  invisible(NULL)
}

.ctb_required <- function(object, field, context) {
  value <- object[[field]]
  if (is.null(value)) {
    .ctb_stop(sprintf("%s is missing %s", context, field))
  }
  value
}

.ctb_object <- function(object, field, context) {
  value <- .ctb_required(object, field, context)
  if (!.ctb_is_object(value)) {
    .ctb_stop(sprintf("%s.%s must be an object", context, field))
  }
  value
}

.ctb_array <- function(object, field, context) {
  value <- .ctb_required(object, field, context)
  if (!.ctb_is_array(value)) {
    .ctb_stop(sprintf("%s.%s must be an array", context, field))
  }
  value
}

.ctb_text <- function(object, field, context) {
  value <- .ctb_required(object, field, context)
  if (!is.character(value) || length(value) != 1L || is.na(value)) {
    .ctb_stop(sprintf("%s.%s must be a string", context, field))
  }
  value
}

.ctb_boolean <- function(object, field, context) {
  value <- .ctb_required(object, field, context)
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    .ctb_stop(sprintf("%s.%s must be a boolean", context, field))
  }
  value
}

.ctb_integer <- function(object, field, context) {
  value <- .ctb_required(object, field, context)
  if (!is.numeric(value) || is.logical(value) || length(value) != 1L ||
      !is.finite(value) || value != floor(value) ||
      value < -.Machine$integer.max || value > .Machine$integer.max) {
    .ctb_stop(sprintf("%s.%s must be a 32-bit integer", context, field))
  }
  as.integer(value)
}

.ctb_finite <- function(object, field, context) {
  value <- .ctb_required(object, field, context)
  if (!is.numeric(value) || is.logical(value) || length(value) != 1L ||
      !is.finite(value)) {
    .ctb_stop(sprintf("%s.%s must be a finite number", context, field))
  }
  as.numeric(value)
}

.ctb_integer_array <- function(values, context) {
  if (!.ctb_is_array(values)) {
    .ctb_stop(sprintf("%s must be an array", context))
  }
  result <- integer(length(values))
  for (index in seq_along(values)) {
    wrapper <- list(value = values[[index]])
    result[[index]] <- .ctb_integer(wrapper, "value", sprintf("%s[%d]", context, index - 1L))
  }
  result
}

.ctb_bit_array <- function(values, context) {
  if (!.ctb_is_array(values)) {
    .ctb_stop(sprintf("%s must be an array", context))
  }
  result <- integer(length(values))
  for (index in seq_along(values)) {
    value <- values[[index]]
    if (is.logical(value) && length(value) == 1L && !is.na(value)) {
      result[[index]] <- as.integer(value)
    } else if (is.numeric(value) && !is.logical(value) && length(value) == 1L &&
               is.finite(value) && value %in% c(0, 1)) {
      result[[index]] <- as.integer(value)
    } else {
      .ctb_stop(sprintf("%s[%d] must be 0, 1, false, or true", context, index - 1L))
    }
  }
  result
}

.ctb_finite_array <- function(values, context) {
  if (!.ctb_is_array(values)) {
    .ctb_stop(sprintf("%s must be an array", context))
  }
  result <- numeric(length(values))
  for (index in seq_along(values)) {
    wrapper <- list(value = values[[index]])
    result[[index]] <- .ctb_finite(wrapper, "value", sprintf("%s[%d]", context, index - 1L))
  }
  result
}

.ctb_parse_quantization <- function(document, num_features) {
  schema <- .ctb_object(document, "quantization_schema", "predictor")
  bins <- .ctb_integer_array(
    .ctb_array(schema, "num_bins_per_feature", "quantization_schema"),
    "quantization_schema.num_bins_per_feature"
  )
  offsets <- .ctb_integer_array(
    .ctb_array(schema, "cut_offsets", "quantization_schema"),
    "quantization_schema.cut_offsets"
  )
  cuts <- .ctb_finite_array(
    .ctb_array(schema, "cut_values", "quantization_schema"),
    "quantization_schema.cut_values"
  )
  categorical <- .ctb_bit_array(
    .ctb_array(schema, "categorical_mask", "quantization_schema"),
    "quantization_schema.categorical_mask"
  )
  missing <- .ctb_bit_array(
    .ctb_array(schema, "missing_value_mask", "quantization_schema"),
    "quantization_schema.missing_value_mask"
  )
  nan_mode <- .ctb_integer(schema, "nan_mode", "quantization_schema")
  if (!(nan_mode %in% 0:2)) {
    .ctb_stop("quantization_schema.nan_mode must be 0, 1, or 2")
  }
  nan_modes <- if (is.null(schema$nan_modes)) {
    integer()
  } else {
    .ctb_integer_array(schema$nan_modes, "quantization_schema.nan_modes")
  }

  if (length(bins) != num_features || length(categorical) != num_features ||
      length(missing) != num_features) {
    .ctb_stop("quantization feature arrays must match num_features")
  }
  if (length(offsets) != as.double(num_features) + 1) {
    .ctb_stop("cut_offsets length must be num_features + 1")
  }
  if (!(length(nan_modes) %in% c(0L, num_features))) {
    .ctb_stop("nan_modes must be empty or match num_features")
  }
  if (offsets[[1L]] != 0L || offsets[[length(offsets)]] != length(cuts)) {
    .ctb_stop("cut_offsets must start at zero and end at cut_values length")
  }

  for (feature in seq_len(num_features)) {
    if (bins[[feature]] < 0L || bins[[feature]] > 65535L) {
      .ctb_stop("num_bins_per_feature is outside uint16 range")
    }
    begin <- offsets[[feature]]
    end <- offsets[[feature + 1L]]
    if (begin < 0L || begin > end || end > length(cuts)) {
      .ctb_stop("cut_offsets must be monotone and in range")
    }
    feature_nan_mode <- if (length(nan_modes) == 0L) nan_mode else nan_modes[[feature]]
    if (!(feature_nan_mode %in% 0:2)) {
      .ctb_stop("nan_modes entries must be 0, 1, or 2")
    }
    non_missing_bins <- bins[[feature]] - missing[[feature]]
    if (non_missing_bins < 0L) {
      .ctb_stop("missing-value bin count exceeds total bins")
    }
    expected_cuts <- if (categorical[[feature]] != 0L) {
      non_missing_bins
    } else {
      max(non_missing_bins - 1L, 0L)
    }
    if (end - begin != expected_cuts) {
      .ctb_stop("cut count is inconsistent with feature bin metadata")
    }
    if (end - begin > 1L) {
      feature_cuts <- cuts[(begin + 1L):end]
      if (any(diff(feature_cuts) <= 0)) {
        .ctb_stop("feature cuts must be strictly increasing")
      }
    }
  }

  list(
    bins = bins,
    offsets = offsets,
    cuts = cuts,
    categorical = categorical,
    missing = missing,
    nan_mode = nan_mode,
    nan_modes = nan_modes
  )
}

.ctb_parse_tree <- function(tree, quantization, tree_index) {
  if (!.ctb_is_object(tree)) {
    .ctb_stop(sprintf("trees[%d] must be an object", tree_index))
  }
  values <- .ctb_array(tree, "nodes", sprintf("trees[%d]", tree_index))
  if (length(values) == 0L) {
    .ctb_stop(sprintf("trees[%d] must contain nodes", tree_index))
  }
  nodes <- vector("list", length(values))
  for (position in seq_along(values)) {
    value <- values[[position]]
    context <- sprintf("trees[%d].nodes[%d]", tree_index, position - 1L)
    if (!.ctb_is_object(value)) {
      .ctb_stop(sprintf("%s must be an object", context))
    }
    leaf <- .ctb_boolean(value, "is_leaf", context)
    categorical <- .ctb_boolean(value, "is_categorical_split", context)
    feature <- .ctb_integer(value, "split_feature_id", context)
    split_bin <- .ctb_integer(value, "split_bin_index", context)
    left <- .ctb_integer(value, "left_child", context)
    right <- .ctb_integer(value, "right_child", context)
    weight <- .ctb_finite(value, "leaf_weight", context)
    routes <- .ctb_bit_array(
      .ctb_array(value, "left_categories", context),
      sprintf("%s.left_categories", context)
    )

    if (leaf) {
      if (left != -1L || right != -1L) {
        .ctb_stop(sprintf("%s leaf children must be -1", context))
      }
    } else {
      if (feature < 0L || feature >= length(quantization$bins)) {
        .ctb_stop(sprintf("%s split feature is out of range", context))
      }
      if (left < 0L || left >= length(values) || right < 0L ||
          right >= length(values) || left == right) {
        .ctb_stop(sprintf("%s child index is invalid", context))
      }
      feature_position <- feature + 1L
      if (split_bin < 0L || split_bin >= quantization$bins[[feature_position]]) {
        .ctb_stop(sprintf("%s split bin is out of range", context))
      }
      if (categorical) {
        if (quantization$categorical[[feature_position]] == 0L) {
          .ctb_stop(sprintf("%s categorical split uses a numeric feature", context))
        }
        if (length(routes) < quantization$bins[[feature_position]]) {
          .ctb_stop(sprintf("%s categorical routes do not cover all bins", context))
        }
      } else if (quantization$categorical[[feature_position]] != 0L) {
        .ctb_stop(sprintf("%s numeric split uses a categorical feature", context))
      }
    }
    nodes[[position]] <- list(
      leaf = leaf,
      categorical = categorical,
      feature = feature,
      split_bin = split_bin,
      left = left,
      right = right,
      weight = weight,
      routes = routes
    )
  }

  visited <- rep(FALSE, length(nodes))
  pending <- 0L
  while (length(pending) > 0L) {
    node_index <- pending[[length(pending)]]
    pending <- pending[-length(pending)]
    position <- node_index + 1L
    if (visited[[position]]) {
      .ctb_stop(sprintf("trees[%d] contains a cycle or shared child", tree_index))
    }
    visited[[position]] <- TRUE
    node <- nodes[[position]]
    if (!node$leaf) {
      pending <- c(pending, node$right, node$left)
    }
  }
  if (!all(visited)) {
    .ctb_stop(sprintf("trees[%d] contains unreachable nodes", tree_index))
  }
  nodes
}

.ctb_read_bounded_text <- function(path, maximum_bytes) {
  connection <- file(path, open = "rb")
  on.exit(close(connection), add = TRUE)
  chunks <- list()
  total <- 0
  repeat {
    request <- min(65536, floor(maximum_bytes - total) + 1)
    chunk <- readBin(connection, what = "raw", n = as.integer(request))
    if (length(chunk) == 0L) {
      break
    }
    total <- total + length(chunk)
    if (total > maximum_bytes) {
      .ctb_stop("predictor artifact exceeds the configured size limit")
    }
    chunks[[length(chunks) + 1L]] <- chunk
  }
  payload <- if (length(chunks) == 0L) raw() else do.call(c, chunks)
  rawToChar(payload)
}

#' Load a prepared-feature CTBoost JSON predictor
#'
#' @param path Path to a JSON predictor artifact.
#' @param max_artifact_bytes Positive file-size limit, in bytes.
#' @return A locked `ctboost_json_predictor` object.
#' @export
ctboost_load_predictor <- function(path, max_artifact_bytes = 512 * 1024^2) {
  if (!is.character(path) || length(path) != 1L || is.na(path)) {
    .ctb_stop("path must be one non-missing string")
  }
  if (!is.numeric(max_artifact_bytes) || length(max_artifact_bytes) != 1L ||
      !is.finite(max_artifact_bytes) || max_artifact_bytes <= 0) {
    .ctb_stop("max_artifact_bytes must be positive")
  }
  size <- file.info(path)$size
  if (is.na(size)) {
    .ctb_stop("predictor artifact does not exist or is not readable")
  }
  document <- tryCatch(
    jsonlite::parse_json(
      .ctb_read_bounded_text(path, max_artifact_bytes),
      simplifyVector = FALSE
    ),
    error = function(error) .ctb_stop(sprintf("could not parse predictor JSON: %s", error$message))
  )
  if (!.ctb_is_object(document)) {
    .ctb_stop("predictor document must be a JSON object")
  }
  .ctb_reject_duplicate_keys(document)
  format <- .ctb_text(document, "format", "predictor")
  if (format != "ctboost-json-predictor") {
    .ctb_stop(sprintf("unsupported predictor format: %s", format))
  }
  format_version <- .ctb_integer(document, "format_version", "predictor")
  if (!(format_version %in% c(1L, 2L))) {
    .ctb_stop(sprintf("unsupported predictor format version: %d", format_version))
  }
  prepared <- .ctb_boolean(document, "expects_prepared_features", "predictor")
  if (!prepared) {
    .ctb_stop(paste(
      "R inference supports prepared numeric features only;",
      "raw feature_pipeline_state execution is not supported"
    ))
  }

  objective <- .ctb_text(document, "objective_name", "predictor")
  if (nchar(objective) == 0L) {
    .ctb_stop("objective_name must not be empty")
  }
  learning_rate <- .ctb_finite(document, "learning_rate", "predictor")
  prediction_dimension <- .ctb_integer(document, "prediction_dimension", "predictor")
  if (prediction_dimension <= 0L) {
    .ctb_stop("prediction_dimension must be positive")
  }
  normalized_objective <- tolower(trimws(objective))
  binary <- normalized_objective %in% c("logloss", "binary_logloss", "binary:logistic")
  multiclass <- normalized_objective %in% c("multiclass", "softmax", "softmaxloss")
  if (binary && prediction_dimension != 1L) {
    .ctb_stop("binary objectives require prediction_dimension == 1")
  }
  if (multiclass && prediction_dimension < 2L) {
    .ctb_stop("multiclass objectives require prediction_dimension >= 2")
  }
  num_features <- .ctb_integer(document, "num_features", "predictor")
  if (num_features < 0L) {
    .ctb_stop("num_features must be non-negative")
  }
  base_score <- .ctb_finite_array(
    .ctb_array(document, "base_score", "predictor"), "base_score"
  )
  if (length(base_score) != prediction_dimension) {
    .ctb_stop("base_score length must match prediction_dimension")
  }
  tree_learning_rates <- if (is.null(document$tree_learning_rates)) {
    numeric()
  } else {
    .ctb_finite_array(document$tree_learning_rates, "tree_learning_rates")
  }
  quantization <- .ctb_parse_quantization(document, num_features)
  tree_documents <- .ctb_array(document, "trees", "predictor")
  if (length(tree_documents) == 0L) {
    .ctb_stop("predictor must contain at least one tree")
  }
  if (length(tree_documents) %% prediction_dimension != 0L) {
    .ctb_stop("tree count must be divisible by prediction_dimension")
  }
  iteration_count <- length(tree_documents) %/% prediction_dimension
  if (length(tree_learning_rates) > iteration_count) {
    .ctb_stop("tree_learning_rates cannot exceed the iteration count")
  }
  trees <- lapply(seq_along(tree_documents), function(index) {
    .ctb_parse_tree(tree_documents[[index]], quantization, index - 1L)
  })

  labels <- document$class_labels
  if (!is.null(labels)) {
    if (!.ctb_is_array(labels)) {
      .ctb_stop("class_labels must be an array or null")
    }
    expected_labels <- if (binary) 2L else if (multiclass) prediction_dimension else -1L
    if (expected_labels < 0L) {
      .ctb_stop("class_labels are only valid for classification objectives")
    }
    if (length(labels) != expected_labels) {
      .ctb_stop("class_labels length does not match the probability dimension")
    }
  }
  if (!is.null(document$inference_manifest) && !.ctb_is_object(document$inference_manifest)) {
    .ctb_stop("inference_manifest must be an object or null")
  }

  predictor <- list2env(
    list(
      objective_name = objective,
      normalized_objective = normalized_objective,
      learning_rate = learning_rate,
      tree_learning_rates = tree_learning_rates,
      base_score = base_score,
      prediction_dimension = prediction_dimension,
      num_features = num_features,
      quantization = quantization,
      trees = trees,
      class_labels = labels,
      inference_manifest = document$inference_manifest
    ),
    parent = emptyenv()
  )
  class(predictor) <- "ctboost_json_predictor"
  lockEnvironment(predictor, bindings = TRUE)
  predictor
}

.ctb_float32 <- function(values) {
  if (length(values) == 0L) {
    return(as.numeric(values))
  }
  bytes <- writeBin(as.double(values), raw(), size = 4L, endian = .Platform$endian)
  readBin(bytes, what = double(), n = length(values), size = 4L, endian = .Platform$endian)
}

.ctb_lower_bound <- function(values, target) {
  left <- 0L
  right <- length(values)
  while (left < right) {
    middle <- left + (right - left) %/% 2L
    if (values[[middle + 1L]] < target) {
      left <- middle + 1L
    } else {
      right <- middle
    }
  }
  left
}

.ctb_upper_bound <- function(values, target) {
  left <- 0L
  right <- length(values)
  while (left < right) {
    middle <- left + (right - left) %/% 2L
    if (target < values[[middle + 1L]]) {
      right <- middle
    } else {
      left <- middle + 1L
    }
  }
  left
}

.ctb_bin_value <- function(schema, feature, value) {
  bins <- schema$bins[[feature]]
  if (bins == 0L) {
    return(0L)
  }
  nan_mode <- if (length(schema$nan_modes) == 0L) {
    schema$nan_mode
  } else {
    schema$nan_modes[[feature]]
  }
  if (is.na(value)) {
    return(if (nan_mode == 2L) bins - 1L else 0L)
  }
  non_missing_bins <- bins - schema$missing[[feature]]
  if (non_missing_bins == 0L) {
    return(if (nan_mode == 2L) bins - 1L else 0L)
  }
  offset <- if (schema$missing[[feature]] != 0L && nan_mode == 1L) 1L else 0L
  begin <- schema$offsets[[feature]]
  end <- schema$offsets[[feature + 1L]]
  cuts <- if (end == begin) numeric() else schema$cuts[(begin + 1L):end]
  if (schema$categorical[[feature]] != 0L) {
    insertion <- .ctb_lower_bound(cuts, value)
    return(offset + min(insertion, non_missing_bins - 1L))
  }
  offset + .ctb_upper_bound(cuts, value)
}

.ctb_score_row <- function(object, row) {
  bins <- integer(object$num_features)
  for (feature in seq_len(object$num_features)) {
    bins[[feature]] <- .ctb_bin_value(object$quantization, feature, row[[feature]])
  }
  scores <- object$base_score
  for (tree_index in seq_along(object$trees)) {
    nodes <- object$trees[[tree_index]]
    node_index <- 0L
    leaf_weight <- NULL
    for (step in seq_along(nodes)) {
      node <- nodes[[node_index + 1L]]
      if (node$leaf) {
        leaf_weight <- node$weight
        break
      }
      bin <- bins[[node$feature + 1L]]
      go_left <- if (node$categorical) {
        node$routes[[bin + 1L]] != 0L
      } else {
        bin <= node$split_bin
      }
      node_index <- if (go_left) node$left else node$right
    }
    if (is.null(leaf_weight)) {
      .ctb_stop("validated tree traversal exceeded its node count")
    }
    iteration <- (tree_index - 1L) %/% object$prediction_dimension
    scale <- if (iteration < length(object$tree_learning_rates)) {
      object$tree_learning_rates[[iteration + 1L]]
    } else {
      object$learning_rate
    }
    output <- (tree_index - 1L) %% object$prediction_dimension + 1L
    scores[[output]] <- scores[[output]] + scale * leaf_weight
  }
  scores
}

.ctb_input_matrix <- function(object, newdata) {
  single <- is.null(dim(newdata))
  if (single) {
    if (!is.numeric(newdata) || length(newdata) != object$num_features) {
      .ctb_stop(sprintf("expected one row with %d numeric features", object$num_features))
    }
    values <- matrix(as.numeric(newdata), nrow = 1L)
  } else {
    if (is.data.frame(newdata)) {
      if (!all(vapply(newdata, is.numeric, logical(1L)))) {
        .ctb_stop("prepared data-frame columns must all be numeric")
      }
      newdata <- as.matrix(newdata)
    }
    if (!is.matrix(newdata) || !is.numeric(newdata)) {
      .ctb_stop("newdata must be a numeric vector, matrix, or data frame")
    }
    if (ncol(newdata) != object$num_features) {
      .ctb_stop(sprintf("expected %d features, got %d", object$num_features, ncol(newdata)))
    }
    values <- newdata
  }
  rounded <- .ctb_float32(as.vector(values))
  dim(rounded) <- dim(values)
  list(values = rounded, single = single)
}

#' Score raw CTBoost margins
#' @param object A predictor returned by [ctboost_load_predictor()].
#' @param newdata Prepared numeric vector, matrix, or data frame.
#' @export
ctboost_predict_raw <- function(object, newdata) {
  if (!inherits(object, "ctboost_json_predictor")) {
    .ctb_stop("object must be a ctboost_json_predictor")
  }
  input <- .ctb_input_matrix(object, newdata)
  result <- matrix(0, nrow = nrow(input$values), ncol = object$prediction_dimension)
  for (row in seq_len(nrow(input$values))) {
    result[row, ] <- .ctb_score_row(object, input$values[row, ])
  }
  if (input$single) {
    return(if (object$prediction_dimension == 1L) result[[1L]] else as.numeric(result[1L, ]))
  }
  if (object$prediction_dimension == 1L) as.numeric(result[, 1L]) else result
}

.ctb_sigmoid <- function(value) {
  if (value >= 0) {
    exponential <- exp(-value)
    1 / (1 + exponential)
  } else {
    exponential <- exp(value)
    exponential / (1 + exponential)
  }
}

#' Score CTBoost classification probabilities
#' @inheritParams ctboost_predict_raw
#' @export
ctboost_predict_proba <- function(object, newdata) {
  raw <- ctboost_predict_raw(object, newdata)
  binary <- object$normalized_objective %in% c("logloss", "binary_logloss", "binary:logistic")
  multiclass <- object$normalized_objective %in% c("multiclass", "softmax", "softmaxloss")
  single <- is.null(dim(newdata))
  if (binary) {
    values <- if (single) raw else as.numeric(raw)
    probabilities <- t(vapply(values, function(value) {
      positive <- .ctb_sigmoid(value)
      c(1 - positive, positive)
    }, numeric(2L)))
    return(if (single) as.numeric(probabilities[1L, ]) else probabilities)
  }
  if (multiclass) {
    scores <- if (single) matrix(raw, nrow = 1L) else raw
    probabilities <- matrix(0, nrow = nrow(scores), ncol = ncol(scores))
    for (row in seq_len(nrow(scores))) {
      shifted <- scores[row, ] - max(scores[row, ])
      exponentials <- exp(shifted)
      probabilities[row, ] <- exponentials / sum(exponentials)
    }
    return(if (single) as.numeric(probabilities[1L, ]) else probabilities)
  }
  .ctb_stop("probability prediction is only available for classification objectives")
}

#' Score CTBoost class labels
#' @inheritParams ctboost_predict_raw
#' @export
ctboost_predict_class <- function(object, newdata) {
  probabilities <- ctboost_predict_proba(object, newdata)
  single <- is.null(dim(newdata))
  probability_matrix <- if (single) matrix(probabilities, nrow = 1L) else probabilities
  indices <- max.col(probability_matrix, ties.method = "first") - 1L
  if (is.null(object$class_labels)) {
    return(if (single) indices[[1L]] else indices)
  }
  labels <- lapply(indices, function(index) object$class_labels[[index + 1L]])
  if (single) {
    return(labels[[1L]])
  }
  atomic_scalar <- vapply(
    object$class_labels,
    function(label) !is.null(label) && is.atomic(label) && length(label) == 1L,
    logical(1L)
  )
  label_kind <- vapply(
    object$class_labels,
    function(label) {
      if (is.null(label) || !is.atomic(label) || length(label) != 1L) {
        return("complex")
      }
      if (is.integer(label) || is.double(label)) "numeric" else typeof(label)
    },
    character(1L)
  )
  if (all(atomic_scalar) && length(unique(label_kind)) == 1L) {
    return(unlist(labels, recursive = FALSE, use.names = FALSE))
  }
  labels
}

#' Predict with a CTBoost JSON predictor
#' @param object A predictor returned by [ctboost_load_predictor()].
#' @param newdata Prepared numeric vector, matrix, or data frame.
#' @param type One of `raw`, `probability`, or `class`.
#' @param ... Reserved for S3 compatibility.
#' @export
predict.ctboost_json_predictor <- function(
    object,
    newdata,
    type = c("raw", "probability", "class"),
    ...) {
  type <- match.arg(type)
  switch(
    type,
    raw = ctboost_predict_raw(object, newdata),
    probability = ctboost_predict_proba(object, newdata),
    class = ctboost_predict_class(object, newdata)
  )
}
