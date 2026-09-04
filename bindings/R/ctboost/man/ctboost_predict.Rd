\name{ctboost_predict}
\alias{ctboost_predict_raw}
\alias{ctboost_predict_proba}
\alias{ctboost_predict_class}
\alias{predict.ctboost_json_predictor}
\title{Score with a prepared-feature CTBoost JSON predictor}
\usage{
ctboost_predict_raw(object, newdata)

ctboost_predict_proba(object, newdata)

ctboost_predict_class(object, newdata)

\method{predict}{ctboost_json_predictor}(
  object,
  newdata,
  type = c("raw", "probability", "class"),
  ...
)
}
\arguments{
  \item{object}{A predictor returned by \code{ctboost_load_predictor()}.}
  \item{newdata}{A prepared numeric vector, matrix, or data frame.}
  \item{type}{The prediction type: raw margin, probability, or class.}
  \item{...}{Reserved for S3 compatibility.}
}
\value{
Raw margins, class probabilities, or class labels for the supplied rows.
}
\description{
Scores prepared numeric features without executing raw-feature preprocessing.
}
