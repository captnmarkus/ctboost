"""Shared prediction rule for the frozen multiclass calibration experiment."""


def apply_temperature(payload, raw_margins):
    import numpy as np
    temperature = payload.get("temperature")
    if temperature is None:
        return raw_margins
    if payload["problem"] != "multiclass":
        raise ValueError("Temperature scaling is restricted to multiclass margins")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    margins = np.asarray(raw_margins, dtype=np.float64)
    if margins.ndim != 2 or not np.isfinite(margins).all():
        raise ValueError("Expected finite two-dimensional multiclass margins")
    return margins / temperature
