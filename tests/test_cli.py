import csv
from importlib import metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

import ctboost
from ctboost.cli import (
    CLIError,
    _load_any_model,
    _load_dataset,
    _load_json_params,
    _normalize_group_id,
    build_parser,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT) + (
        os.pathsep + current_pythonpath if current_pythonpath else ""
    )
    return subprocess.run(
        [sys.executable, "-m", "ctboost", *map(str, arguments)],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _write_regression_csv(path: Path, row_count: int = 48) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["value", "category", "target"])
        for index in range(row_count):
            value = (index - row_count / 2.0) / 8.0
            category = ("red", "green", "blue")[index % 3]
            target = 1.5 * value + (1.0 if category == "red" else -0.25)
            writer.writerow([format(value, ".8g"), category, format(target, ".8g")])


def test_parser_and_json_parameter_sources(tmp_path: Path):
    parsed = build_parser().parse_args(
        [
            "predict",
            "--model",
            "model.ctb",
            "--input",
            "features.npy",
            "--output",
            "predictions.npy",
            "--prediction-type",
            "probability",
        ]
    )
    assert parsed.command == "predict"
    assert parsed.prediction_type == "probability"

    parameter_path = tmp_path / "parameters.json"
    parameter_path.write_text('{"iterations": 7, "random_seed": 19}', encoding="utf-8")
    assert _load_json_params(str(parameter_path)) == {"iterations": 7, "random_seed": 19}
    assert _load_json_params("@" + str(parameter_path)) == {
        "iterations": 7,
        "random_seed": 19,
    }
    assert _load_json_params('{"max_depth": 2}') == {"max_depth": 2}
    with pytest.raises(CLIError, match="JSON parameters must decode to an object"):
        _load_json_params("[1, 2]")


def test_csv_train_predict_inspect_manifest_and_no_overwrite(tmp_path: Path):
    training_path = tmp_path / "training.csv"
    model_path = tmp_path / "regressor.ctb"
    _write_regression_csv(training_path)

    trained = _run_cli(
        tmp_path,
        "train",
        "--task",
        "regression",
        "--input",
        training_path,
        "--target",
        "target",
        "--categorical",
        "category",
        "--model",
        model_path,
        "--params",
        '{"iterations": 6, "learning_rate": 0.2, "max_depth": 2, "random_seed": 13}',
    )
    assert trained.returncode == 0, trained.stderr
    training_summary = json.loads(trained.stdout)
    assert training_summary["command"] == "train"
    assert training_summary["model_type"] == "CTBoostRegressor"
    assert training_summary["rows"] == 48
    assert training_summary["feature_columns"] == ["value", "category"]
    assert model_path.is_file()

    inspected = _run_cli(tmp_path, "inspect", "--model", model_path)
    assert inspected.returncode == 0, inspected.stderr
    inspection = json.loads(inspected.stdout)
    assert inspection["model_type"] == "CTBoostRegressor"
    assert inspection["objective"] == "RMSE"
    assert inspection["iterations_trained"] == 6
    assert inspection["inference_manifest"]["artifact"]["estimator"] == "CTBoostRegressor"

    first_output = tmp_path / "predictions.csv"
    first = _run_cli(
        tmp_path,
        "predict",
        "--model",
        model_path,
        "--input",
        training_path,
        "--drop-column",
        "target",
        "--output",
        first_output,
    )
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["shape"] == [48]
    first_bytes = first_output.read_bytes()
    assert first_bytes.startswith(b"raw\n")

    second_output = tmp_path / "predictions-second.csv"
    second = _run_cli(
        tmp_path,
        "predict",
        "--model",
        model_path,
        "--input",
        training_path,
        "--drop-column",
        "target",
        "--output",
        second_output,
    )
    assert second.returncode == 0, second.stderr
    assert second_output.read_bytes() == first_bytes

    protected = _run_cli(
        tmp_path,
        "predict",
        "--model",
        model_path,
        "--input",
        training_path,
        "--drop-column",
        "target",
        "--output",
        first_output,
    )
    assert protected.returncode == 2
    assert "output already exists" in protected.stderr
    assert first_output.read_bytes() == first_bytes

    manifest_path = tmp_path / "manifest.json"
    exported = _run_cli(
        tmp_path,
        "export",
        "--model",
        model_path,
        "--format",
        "manifest",
        "--output",
        manifest_path,
    )
    assert exported.returncode == 0, exported.stderr
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["format"] == (
        "ctboost-inference-manifest"
    )


def test_npz_classification_probability_class_and_npy_input(tmp_path: Path):
    rng = np.random.default_rng(31)
    features = rng.normal(size=(64, 4)).astype(np.float32)
    target = (features[:, 0] - 0.4 * features[:, 1] > 0.0).astype(np.int64)
    archive_path = tmp_path / "classification.npz"
    np.savez(archive_path, X=features, target=target)
    model_path = tmp_path / "classifier.ctb"

    trained = _run_cli(
        tmp_path,
        "train",
        "--task",
        "classification",
        "--input",
        archive_path,
        "--array-key",
        "X",
        "--target",
        "target",
        "--model",
        model_path,
        "--iterations",
        "6",
        "--max-depth",
        "2",
        "--random-seed",
        "23",
    )
    assert trained.returncode == 0, trained.stderr
    assert json.loads(trained.stdout)["class_labels"] == [0, 1]

    probability_path = tmp_path / "probabilities.npy"
    probability = _run_cli(
        tmp_path,
        "predict",
        "--model",
        model_path,
        "--input",
        archive_path,
        "--array-key",
        "X",
        "--prediction-type",
        "probability",
        "--output",
        probability_path,
    )
    assert probability.returncode == 0, probability.stderr
    probability_values = np.load(probability_path, allow_pickle=False)
    assert probability_values.shape == (64, 2)
    np.testing.assert_allclose(probability_values.sum(axis=1), 1.0, atol=1e-6)

    feature_path = tmp_path / "features.npy"
    np.save(feature_path, features, allow_pickle=False)
    class_path = tmp_path / "classes.json"
    classified = _run_cli(
        tmp_path,
        "predict",
        "--model",
        model_path,
        "--input",
        feature_path,
        "--prediction-type",
        "class",
        "--output",
        class_path,
    )
    assert classified.returncode == 0, classified.stderr
    classes = json.loads(class_path.read_text(encoding="utf-8"))
    assert len(classes) == 64
    assert set(classes) <= {0, 1}


def test_info_output_and_actionable_input_error(tmp_path: Path):
    info_path = tmp_path / "info.json"
    result = _run_cli(tmp_path, "info", "--output", info_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["command"] == "info"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["ctboost_version"]
    assert info["build"]["version"] == info["ctboost_version"]

    unsupported = tmp_path / "features.txt"
    unsupported.write_text("1,2\n", encoding="utf-8")
    failed = _run_cli(
        tmp_path,
        "predict",
        "--model",
        tmp_path / "missing.ctb",
        "--input",
        unsupported,
        "--output",
        tmp_path / "predictions.csv",
    )
    assert failed.returncode == 2
    assert "model file does not exist" in failed.stderr
    assert "Traceback" not in failed.stderr

    untrusted_model = tmp_path / "untrusted.pkl"
    untrusted_model.write_bytes(b"\x80\x04not-a-trusted-model")
    refused = _run_cli(
        tmp_path,
        "inspect",
        "--model",
        untrusted_model,
    )
    assert refused.returncode == 2
    assert "pickle may execute arbitrary code" in refused.stderr
    assert "--allow-unsafe-pickle" in refused.stderr


@pytest.mark.parametrize("suffix", ["parquet", "feather"])
def test_optional_arrow_table_input(tmp_path: Path, suffix: str):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / ("features." + suffix)
    frame = pd.DataFrame({"score": [1.0, 2.0], "segment": ["a", "b"]})
    if suffix == "feather":
        frame.to_feather(path)
    else:
        frame.to_parquet(path, index=False)

    loaded = _load_dataset(str(path))
    assert loaded.columns == ["score", "segment"]
    assert loaded.values.shape == (2, 2)


def test_project_registers_console_script_and_cli_extra():
    project_path = REPOSITORY_ROOT / "pyproject.toml"
    if project_path.is_file():
        project = project_path.read_text(encoding="utf-8")
        assert 'ctboost = "ctboost.cli:main"' in project
        assert "cli = [" in project
        assert '"pandas>=1.5"' in project
        assert '"pyarrow>=12"' in project
        return

    distribution = metadata.distribution("ctboost")
    console_scripts = {
        entry_point.name: entry_point.value
        for entry_point in distribution.entry_points
        if entry_point.group == "console_scripts"
    }
    assert console_scripts["ctboost"] == "ctboost.cli:main"

    requirements = distribution.requires or []
    cli_requirements = [
        requirement
        for requirement in requirements
        if 'extra == "cli"' in requirement
    ]
    assert any(
        requirement.startswith("pandas>=1.5") for requirement in cli_requirements
    )
    assert any(
        requirement.startswith("pyarrow>=12") for requirement in cli_requirements
    )


def test_group_id_normalization_does_not_truncate_distinct_float_identifiers():
    encoded = _normalize_group_id(np.asarray([1.2, 1.8, 1.2, 4.0]))
    assert encoded[0] == encoded[2]
    assert encoded[0] != encoded[1]
    assert len(np.unique(encoded)) == 3


@pytest.mark.parametrize(
    ("suffix", "model_format", "message"),
    [
        ("pkl", "json", "cannot use a pickle filename"),
        ("ctb", "pickle", "cannot use a JSON model filename"),
    ],
)
def test_train_rejects_serialization_format_suffixes_that_cannot_be_reloaded(
    tmp_path: Path,
    suffix: str,
    model_format: str,
    message: str,
):
    training_path = tmp_path / "training.csv"
    _write_regression_csv(training_path, row_count=8)
    result = _run_cli(
        tmp_path,
        "train",
        "--input",
        training_path,
        "--target",
        "target",
        "--model",
        tmp_path / ("model." + suffix),
        "--model-format",
        model_format,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_train_rejects_ignored_group_and_key_arguments(tmp_path: Path):
    training_path = tmp_path / "training.csv"
    _write_regression_csv(training_path, row_count=8)
    ignored_group = _run_cli(
        tmp_path,
        "train",
        "--input",
        training_path,
        "--target",
        "target",
        "--group",
        "value",
        "--model",
        tmp_path / "model.ctb",
    )
    assert ignored_group.returncode == 2
    assert "only valid for ranking" in ignored_group.stderr

    ignored_key = _run_cli(
        tmp_path,
        "train",
        "--input",
        training_path,
        "--target",
        "target",
        "--target-key",
        "target",
        "--model",
        tmp_path / "model-key.ctb",
    )
    assert ignored_key.returncode == 2
    assert "--target-key requires --target-file" in ignored_key.stderr


def test_trusted_estimator_pickle_is_deserialized_exactly_once(tmp_path: Path, monkeypatch):
    features = np.arange(24, dtype=np.float32).reshape(12, 2)
    target = features[:, 0] - 0.5 * features[:, 1]
    model = ctboost.CTBoostRegressor(iterations=2, max_depth=1, alpha=1.0).fit(
        features,
        target,
    )
    path = tmp_path / "trusted.pkl"
    model.save_model(path, model_format="pickle")

    original_load = pickle.load
    calls = []

    def counted_load(stream, *args, **kwargs):
        calls.append(stream.name)
        return original_load(stream, *args, **kwargs)

    monkeypatch.setattr(pickle, "load", counted_load)
    loaded, model_type = _load_any_model(path, allow_unsafe_pickle=True)

    assert model_type == "CTBoostRegressor"
    assert len(calls) == 1
    np.testing.assert_array_equal(loaded.predict(features), model.predict(features))
