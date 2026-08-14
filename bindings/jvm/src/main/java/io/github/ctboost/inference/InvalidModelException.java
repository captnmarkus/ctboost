package io.github.ctboost.inference;

/** Raised when a JSON document is not a safe, supported CTBoost predictor. */
public final class InvalidModelException extends IllegalArgumentException {
    public InvalidModelException(String message) {
        super(message);
    }

    public InvalidModelException(String message, Throwable cause) {
        super(message, cause);
    }
}
