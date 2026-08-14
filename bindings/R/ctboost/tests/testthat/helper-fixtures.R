ctboost_fixture <- function(name) {
  override <- Sys.getenv("CTBOOST_CONFORMANCE_DIR", unset = "")
  if (nzchar(override)) {
    return(normalizePath(file.path(override, name), mustWork = TRUE))
  }

  directory <- normalizePath(getwd(), mustWork = TRUE)
  repeat {
    candidate <- file.path(directory, "tests", "export_conformance")
    if (file.exists(file.path(candidate, "prepared_regression_v1.json"))) {
      return(normalizePath(file.path(candidate, name), mustWork = TRUE))
    }
    parent <- dirname(directory)
    if (identical(parent, directory)) {
      break
    }
    directory <- parent
  }

  packaged <- system.file(
    "extdata", "export_conformance", name, package = "ctboost"
  )
  if (nzchar(packaged) && file.exists(packaged)) {
    return(normalizePath(packaged, mustWork = TRUE))
  }
  stop(
    "could not locate packaged or repository export-conformance fixtures",
    call. = FALSE
  )
}

ctboost_cases <- function(name) {
  jsonlite::fromJSON(ctboost_fixture(name), simplifyVector = FALSE)
}

ctboost_rows <- function(values) {
  rows <- lapply(values, function(row) {
    vapply(row, function(value) {
      if (is.character(value) && identical(value, "NaN")) NaN else as.numeric(value)
    }, numeric(1L))
  })
  do.call(rbind, rows)
}
