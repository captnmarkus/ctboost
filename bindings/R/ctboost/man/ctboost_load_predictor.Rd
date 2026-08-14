\name{ctboost_load_predictor}
\alias{ctboost_load_predictor}
\title{Load a prepared-feature CTBoost JSON predictor}
\usage{
ctboost_load_predictor(path, max_artifact_bytes = 512 * 1024^2)
}
\arguments{
  \item{path}{Path to a CTBoost JSON predictor artifact.}
  \item{max_artifact_bytes}{Positive maximum artifact size in bytes.}
}
\value{
An object of class \code{ctboost_json_predictor}.
}
\description{
Loads and validates a CTBoost JSON predictor that expects already-prepared
numeric features. Artifacts that request raw-feature preprocessing are rejected.
}
