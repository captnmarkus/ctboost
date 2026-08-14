import json
from pathlib import Path

import numpy as np
import pytest

from ctboost.export_runtime import ExportedPredictor


_FIXTURE_ROOT = Path(__file__).parent / "export_conformance"
_SCHEMA_PATH = Path(__file__).parents[1] / "spec" / "json-predictor-prepared.schema.json"


@pytest.mark.parametrize(
    "case_name",
    ("prepared_regression_v1", "prepared_binary_v2", "prepared_multiclass_v2"),
)
def test_python_prepared_profile_schema_and_golden_outputs(case_name):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    artifact = json.loads(
        (_FIXTURE_ROOT / f"{case_name}.json").read_text(encoding="utf-8")
    )
    cases = json.loads(
        (_FIXTURE_ROOT / f"{case_name}.cases.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(artifact)

    predictor = ExportedPredictor(artifact)
    raw = np.asarray(predictor.predict_raw(cases["rows"]), dtype=np.float64)
    np.testing.assert_allclose(
        raw,
        np.asarray(cases["raw_predictions"], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )
    if "probabilities" in cases:
        probabilities = np.asarray(
            predictor.predict_proba(cases["rows"]), dtype=np.float64
        )
        np.testing.assert_allclose(
            probabilities,
            np.asarray(cases["probabilities"], dtype=np.float64),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_array_equal(
            np.argmax(probabilities, axis=1),
            np.asarray(cases["class_indices"], dtype=np.int64),
        )
        assert predictor.predict_class(cases["rows"]) == cases["class_labels"]
