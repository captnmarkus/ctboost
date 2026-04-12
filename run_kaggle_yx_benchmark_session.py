from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

from kagglehub import auth as kaggle_auth
from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
    ApiSaveKernelRequest,
)
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

from run_kaggle_kernel_session import (
    PATCHED_FILES,
    _download_outputs,
    _load_overlay_payloads,
    _poll_kernel,
)


def _build_kernel_source(
    repo_root: Path,
    base_commit: str,
    max_folds: int,
    include_optuna: bool,
) -> str:
    overlay_json = json.dumps(_load_overlay_payloads(repo_root), separators=(",", ":"))
    patched_files_json = json.dumps(PATCHED_FILES)
    return textwrap.dedent(
        f"""
        import base64
        import json
        import os
        import random
        import shutil
        import subprocess
        import sys
        import time
        import warnings
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from sklearn.model_selection import KFold, StratifiedKFold

        PATCHED_FILES = {patched_files_json}
        OVERLAY = json.loads({overlay_json!r})
        BASE_COMMIT = {base_commit!r}
        MAX_FOLDS = {max_folds}
        INCLUDE_OPTUNA = {include_optuna!r}
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)

        SEED = 2026
        FOLD_SEED = 42
        FOLD_KIND = "kfold"
        N_FOLDS = 5
        TE_ALPHA = 1.0
        TE_REPEATS = 4
        WEIGHT_OPTUNA_TRIALS = 200
        EARLY_STOPPING_ROUNDS = 100
        RESAMPLE_BY_WEIGHT = False
        RESAMPLE_FACTOR = 1.0
        CTBOOST_PARAMS = {{
            "iterations": 900,
            "learning_rate": 0.05,
            "max_depth": 4,
            "alpha": 1.0,
            "lambda_l2": 5.0,
        }}


        def run(cmd, cwd=None, env=None, check=True):
            print("+", " ".join(cmd), flush=True)
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
            )
            if completed.stdout:
                print(completed.stdout, flush=True)
            if completed.stderr:
                print(completed.stderr, flush=True)
            if check and completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    cmd,
                    output=completed.stdout,
                    stderr=completed.stderr,
                )
            return completed


        def log_event(event, **payload):
            record = {{"event": event, **payload}}
            print(json.dumps(record), flush=True)


        def seed_everything(seed):
            np.random.seed(seed)
            random.seed(seed)


        def accuracy_score(targets, predictions):
            predicted = predictions
            if len(predicted.shape) == 2:
                predicted = np.argmax(predicted, axis=1)
            classes = 3
            balanced_accuracy = 0.0
            for class_index in range(classes):
                balanced_accuracy += (
                    np.sum((targets == class_index) & (predicted == class_index))
                    / np.sum(targets == class_index)
                    / classes
                )
            return balanced_accuracy


        class OrderedTE:
            def __init__(self, a=1.0):
                self.a = a

            def fit(self, train, category_cols=None, target_col="target"):
                if category_cols is None:
                    category_cols = []
                self.train = train
                self.target_col = target_col
                self.category_cols = category_cols
                self.classes_ = sorted(train[target_col].unique())
                self.num_classes_ = len(self.classes_)
                self.global_prior_ = (
                    train[target_col].value_counts(normalize=True).sort_index().values
                )
                te_frames = []

                for column in self.category_cols:
                    column_te = {{}}
                    stats_list = []
                    for class_offset, class_index in enumerate(self.classes_):
                        y_binary = (train[target_col] == class_index).astype(int)
                        df = train[[column]].copy()
                        df["y"] = y_binary.values
                        df["cnt"] = 1
                        df["cum_cnt"] = df.groupby(column)["cnt"].cumsum() - df["cnt"]
                        df["cum_sum"] = df.groupby(column)["y"].cumsum() - df["y"]
                        smooth_prior = self.a * self.global_prior_[class_offset]
                        te_col = f"{{column}}_TE_cls{{class_index}}"
                        te_values = (df["cum_sum"] + smooth_prior) / (df["cum_cnt"] + self.a)
                        te_values = te_values.mask(
                            df["cum_cnt"] == -1, self.global_prior_[class_offset]
                        )
                        column_te[te_col] = te_values.astype(np.float32)

                        stats_df = df.groupby(column)["y"].agg(["count", "sum"])
                        stats_df.columns = [
                            f"{{column}}_count_cls{{class_index}}",
                            f"{{column}}_sum_cls{{class_index}}",
                        ]
                        stats_df[f"{{column}}_prior_cls{{class_index}}"] = self.global_prior_[
                            class_offset
                        ]
                        stats_list.append(stats_df)

                    te_frames.append(pd.DataFrame(column_te, index=train.index))
                    setattr(self, f"{{column}}_stats", pd.concat(stats_list, axis=1))

                if te_frames:
                    self.train = pd.concat([self.train] + te_frames, axis=1, copy=False)

                return self.train

            def transform(self, test):
                te_frames = []
                for column in self.category_cols:
                    stats_df = getattr(self, f"{{column}}_stats")
                    lookup = test[[column]].join(stats_df, on=column)
                    column_te = {{}}
                    for class_offset, class_index in enumerate(self.classes_):
                        te_col = f"{{column}}_TE_cls{{class_index}}"
                        count_col = f"{{column}}_count_cls{{class_index}}"
                        sum_col = f"{{column}}_sum_cls{{class_index}}"
                        prior_col = f"{{column}}_prior_cls{{class_index}}"
                        if count_col in lookup.columns:
                            te_values = (
                                lookup[sum_col] + self.a * lookup[prior_col]
                            ) / (lookup[count_col] + self.a)
                            column_te[te_col] = te_values.fillna(
                                lookup[prior_col]
                            ).astype(np.float32)
                        else:
                            column_te[te_col] = np.full(
                                len(test), self.global_prior_[class_offset], dtype=np.float32
                            )
                    te_frames.append(pd.DataFrame(column_te, index=test.index))

                if not te_frames:
                    return test
                return pd.concat([test] + te_frames, axis=1, copy=False)


        def reduce_mem_usage(df, float16_as32=True):
            for column in df.columns:
                column_type = df[column].dtype
                if column_type == object or str(column_type) == "category":
                    continue
                column_min = df[column].min()
                column_max = df[column].max()
                if str(column_type).startswith("int"):
                    if column_min > np.iinfo(np.int8).min and column_max < np.iinfo(np.int8).max:
                        df[column] = df[column].astype(np.int8)
                    elif column_min > np.iinfo(np.int16).min and column_max < np.iinfo(np.int16).max:
                        df[column] = df[column].astype(np.int16)
                    elif column_min > np.iinfo(np.int32).min and column_max < np.iinfo(np.int32).max:
                        df[column] = df[column].astype(np.int32)
                else:
                    if (
                        column_min > np.finfo(np.float16).min
                        and column_max < np.finfo(np.float16).max
                    ):
                        df[column] = df[column].astype(np.float32 if float16_as32 else np.float16)
                    elif (
                        column_min > np.finfo(np.float32).min
                        and column_max < np.finfo(np.float32).max
                    ):
                        df[column] = df[column].astype(np.float32)
                    else:
                        df[column] = df[column].astype(np.float64)
            return df


        def maybe_weight_resample(X_train, y_train, train_weights, fold_seed):
            if not RESAMPLE_BY_WEIGHT:
                return X_train, y_train, train_weights
            probabilities = np.asarray(train_weights, dtype=np.float64)
            probabilities = probabilities / probabilities.sum()
            sample_size = max(len(X_train), int(round(len(X_train) * RESAMPLE_FACTOR)))
            rng = np.random.default_rng(fold_seed)
            sampled_idx = rng.choice(
                len(X_train), size=sample_size, replace=True, p=probabilities
            )
            X_resampled = X_train.iloc[sampled_idx].reset_index(drop=True)
            y_resampled = y_train.iloc[sampled_idx].reset_index(drop=True)
            w_resampled = np.asarray(train_weights, dtype=np.float32)[sampled_idx]
            return X_resampled, y_resampled, w_resampled


        repo = Path("/kaggle/working/ctboost")
        build_start = time.perf_counter()
        if repo.exists():
            shutil.rmtree(repo)
        run(["git", "clone", "https://github.com/captnmarkus/ctboost.git", str(repo)])
        run(["git", "checkout", BASE_COMMIT], cwd=repo)
        for relative_path, encoded in OVERLAY.items():
            destination = repo / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(encoded))

        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "ctboost"], check=False)
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "build",
                "cmake",
                "ninja",
                "numpy",
                "pandas",
                "optuna",
                "pybind11",
                "scikit-build-core",
                "scikit-learn",
            ]
        )
        build_env = os.environ.copy()
        build_env["CMAKE_ARGS"] = "-DCTBOOST_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75"
        build_env["CUDACXX"] = "/usr/local/cuda/bin/nvcc"
        build_env["CMAKE_CUDA_COMPILER"] = "/usr/local/cuda/bin/nvcc"
        build_env["CUDAToolkit_ROOT"] = "/usr/local/cuda"
        build_env["CUDA_HOME"] = "/usr/local/cuda"
        run([sys.executable, "-m", "pip", "install", "-v", "."], cwd=repo, env=build_env)
        build_seconds = time.perf_counter() - build_start
        log_event("build_complete", seconds=build_seconds)

        import ctboost
        from ctboost import CTBoostClassifier

        warnings.filterwarnings("ignore")
        seed_everything(SEED)
        build_info = ctboost.build_info()
        task_type = "GPU" if build_info.get("cuda_enabled") else "CPU"
        log_event("runtime_ready", ctboost=ctboost.__version__, build_info=build_info, task_type=task_type)

        target_col = "Irrigation_Need"
        data_start = time.perf_counter()
        train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e4/train.csv").drop(
            ["id"], axis=1
        )
        test = pd.read_csv("/kaggle/input/competitions/playground-series-s6e4/test.csv").drop(
            ["id"], axis=1
        )
        target2idx = {{value: index for index, value in enumerate(train[target_col].unique())}}
        idx2target = {{value: index for index, value in target2idx.items()}}
        train[target_col] = train[target_col].map(target2idx)

        cats = [column for column in test.columns if train[column].dtype == object]
        nums = [column for column in test.columns if column not in cats]
        max_values = train[nums].max()

        def fe(df):
            transformed = df
            for column in nums:
                for power in range(-4, 4):
                    transformed[f"{{column}}_digit{{power}}"] = (
                        transformed[column] // (10 ** power) % 10
                    ).astype("int8")
                if max_values[column] < 10:
                    transformed[column] = transformed[column].round(3)
                elif max_values[column] < 100:
                    transformed[column] = transformed[column].round(2)
                else:
                    transformed[column] = transformed[column].round(1)
            return transformed

        train = fe(train)
        test = fe(test)
        drop_cols = [column for column in test.columns if test[column].nunique() == 1]
        train.drop(drop_cols, axis=1, inplace=True)
        test.drop(drop_cols, axis=1, inplace=True)

        category_columns = cats + [column for column in test.columns if "digit" in column]
        for column in category_columns:
            frequency = train[column].value_counts()
            mapping = {{
                value: index
                for index, (value, count) in enumerate(frequency[frequency >= 5].items())
            }}
            default_value = len(mapping)
            train[column] = train[column].map(lambda value: mapping.get(value, default_value))
            test[column] = test[column].map(lambda value: mapping.get(value, default_value))

        features = category_columns + nums
        unique, counts = np.unique(train[target_col].values, return_counts=True)
        count_dict = dict(zip(unique, counts))
        average_count = len(train) / len(unique)
        weights_dict = {{
            class_index: average_count / class_count
            for class_index, class_count in count_dict.items()
        }}
        sample_weights = np.array([weights_dict[label] for label in train[target_col]])

        X = train.drop([target_col], axis=1)
        y = train[target_col]
        test_X = test.copy()
        oof_preds = np.zeros((len(y), 3), dtype=np.float32)
        test_preds = np.zeros((len(test_X), 3), dtype=np.float32)
        log_event(
            "data_ready",
            seconds=time.perf_counter() - data_start,
            train_rows=int(X.shape[0]),
            train_cols=int(X.shape[1]),
            test_rows=int(test_X.shape[0]),
            test_cols=int(test_X.shape[1]),
        )

        params = CTBOOST_PARAMS.copy()
        params["task_type"] = task_type

        if FOLD_KIND == "stratified":
            splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
            split_iter = splitter.split(X, y)
        else:
            splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
            split_iter = splitter.split(X)

        overall_start = time.perf_counter()
        fold_summaries = []
        for fold, (train_idx, val_idx) in enumerate(split_iter):
            if fold >= MAX_FOLDS:
                break

            log_event("fold_start", fold=fold + 1, requested_folds=N_FOLDS)
            fold_start = time.perf_counter()
            preprocess_start = time.perf_counter()

            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()
            train_weights = sample_weights[train_idx]

            te = OrderedTE(a=TE_ALPHA)
            full_df = pd.concat((X_train, y_train), axis=1)
            full_df["weight"] = train_weights
            te_train = pd.concat(
                [
                    te.fit(
                        full_df.sample(frac=1, random_state=FOLD_SEED + repeat_index),
                        category_cols=features,
                        target_col=target_col,
                    )
                    for repeat_index in range(TE_REPEATS)
                ]
            )
            X_train = te_train.drop([target_col, "weight"], axis=1)
            y_train = te_train[target_col]
            train_weights = te_train["weight"].to_numpy()

            X_val = te.transform(X_val)
            X_test = te.transform(test_X.copy())
            X_train.drop(cats, axis=1, inplace=True)
            X_val.drop(cats, axis=1, inplace=True)
            X_test.drop(cats, axis=1, inplace=True)
            X_train = reduce_mem_usage(X_train)
            X_val = reduce_mem_usage(X_val)
            X_test = reduce_mem_usage(X_test)

            X_fit, y_fit, fit_weights = maybe_weight_resample(
                X_train, y_train, train_weights, FOLD_SEED + fold
            )
            preprocess_seconds = time.perf_counter() - preprocess_start
            log_event(
                "fold_preprocess_complete",
                fold=fold + 1,
                preprocess_seconds=preprocess_seconds,
                fit_rows=int(X_fit.shape[0]),
                fit_cols=int(X_fit.shape[1]),
                val_rows=int(X_val.shape[0]),
            )

            model = CTBoostClassifier(**params)
            fit_start = time.perf_counter()
            model.fit(
                X_fit,
                y_fit,
                sample_weight=fit_weights,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            )
            fit_seconds = time.perf_counter() - fit_start
            log_event("fold_fit_complete", fold=fold + 1, fit_seconds=fit_seconds)

            predict_start = time.perf_counter()
            y_pred = model.predict_proba(X_val)
            test_preds += model.predict_proba(X_test) / N_FOLDS
            predict_seconds = time.perf_counter() - predict_start
            oof_preds[val_idx] = y_pred

            fold_summary = {{
                "fold": fold + 1,
                "preprocess_seconds": preprocess_seconds,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fold_seconds": time.perf_counter() - fold_start,
                "best_iteration": None if model.best_iteration_ is None else int(model.best_iteration_),
                "fit_rows": int(X_fit.shape[0]),
                "fit_cols": int(X_fit.shape[1]),
                "validation_rows": int(X_val.shape[0]),
                "validation_score": float(accuracy_score(y_val.to_numpy(), y_pred)),
            }}
            fold_summaries.append(fold_summary)
            log_event("fold_complete", **fold_summary)

        total_seconds = time.perf_counter() - overall_start
        results = {{
            "base_commit": BASE_COMMIT,
            "patched_files": PATCHED_FILES,
            "build_info": build_info,
            "task_type": task_type,
            "build_seconds": build_seconds,
            "max_folds": MAX_FOLDS,
            "requested_folds": N_FOLDS,
            "fold_summaries": fold_summaries,
            "total_seconds": total_seconds,
            "partial_oof_balanced_accuracy": float(accuracy_score(y.to_numpy(), oof_preds)),
        }}

        if INCLUDE_OPTUNA and MAX_FOLDS == N_FOLDS:
            import optuna
            from optuna.samplers import TPESampler

            def objective(trial):
                class_weights = np.array(
                    [
                        trial.suggest_float("cw1", 0.5, 3.0),
                        trial.suggest_float("cw2", 0.5, 3.0),
                        trial.suggest_float("cw3", 0.5, 3.0),
                    ]
                )
                adjusted = oof_preds * class_weights
                adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
                return accuracy_score(y.to_numpy(), np.argmax(adjusted, axis=1))

            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=42),
                study_name="class_weight_optimization",
            )
            study.optimize(objective, n_trials=WEIGHT_OPTUNA_TRIALS, show_progress_bar=False)
            best_cw = np.array(
                [
                    study.best_params["cw1"],
                    study.best_params["cw2"],
                    study.best_params["cw3"],
                ]
            )
            final_oof_probs = oof_preds * best_cw
            final_oof_probs = final_oof_probs / final_oof_probs.sum(axis=1, keepdims=True)
            final_test_probs = test_preds * best_cw
            final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)
            final_test_preds = np.argmax(final_test_probs, axis=1)

            submission = pd.read_csv(
                "/kaggle/input/competitions/playground-series-s6e4/sample_submission.csv"
            )
            submission[target_col] = pd.Series(final_test_preds).map(idx2target)
            submission.to_csv("/kaggle/working/submission.csv", index=False)
            pd.DataFrame(
                {{
                    "p_low": final_oof_probs[:, 0],
                    "p_medium": final_oof_probs[:, 1],
                    "p_high": final_oof_probs[:, 2],
                }}
            ).to_csv("/kaggle/working/oof_predictions.csv", index=False)
            pd.DataFrame(
                {{
                    "p_low": final_test_probs[:, 0],
                    "p_medium": final_test_probs[:, 1],
                    "p_high": final_test_probs[:, 2],
                }}
            ).to_csv("/kaggle/working/test_predictions.csv", index=False)
            results["final_oof_balanced_accuracy"] = float(
                accuracy_score(y.to_numpy(), final_oof_probs)
            )
            results["best_class_weights"] = {{
                "class_0": float(best_cw[0]),
                "class_1": float(best_cw[1]),
                "class_2": float(best_cw[2]),
            }}

        output_path = Path("/kaggle/working/yx_ctboost_benchmark_results.json")
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        """
    ).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME", "maiernator"))
    parser.add_argument("--token", default=os.environ.get("KAGGLE_API_TOKEN"))
    parser.add_argument("--slug", default="ps-s6e4-yx-ctboost-gpu-source-benchmark")
    parser.add_argument("--title", default="PS S6E4 YX CTB Src Bench")
    parser.add_argument("--output-dir", default=".tmp/kaggle-kernel-output-yx-benchmark")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--max-folds", type=int, default=1)
    parser.add_argument("--include-optuna", action="store_true")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("missing Kaggle API token: pass --token or set KAGGLE_API_TOKEN")
    if args.max_folds <= 0:
        raise SystemExit("--max-folds must be positive")

    repo_root = Path(__file__).resolve().parent
    base_commit = os.popen(f'git -C "{repo_root}" rev-parse HEAD').read().strip()
    if not base_commit:
        raise SystemExit("failed to resolve git HEAD")

    kaggle_auth.set_kaggle_api_token(args.token)
    whoami = kaggle_auth.whoami(verbose=False)
    resolved_username = str(whoami.get("username") or args.username)
    print(f"authenticated as {resolved_username}")

    client = KaggleClient(username=resolved_username, api_token=args.token, verbose=False)
    request = ApiSaveKernelRequest()
    request.slug = f"{resolved_username}/{args.slug}"
    request.new_title = args.title
    request.language = "python"
    request.kernel_type = "script"
    request.is_private = True
    request.enable_internet = True
    request.enable_gpu = True
    request.machine_shape = "NvidiaTeslaT4"
    request.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
    request.competition_data_sources = ["playground-series-s6e4"]
    request.session_timeout_seconds = args.timeout_seconds
    request.text = _build_kernel_source(
        repo_root,
        base_commit,
        max_folds=args.max_folds,
        include_optuna=args.include_optuna,
    )

    response = client.kernels.kernels_api_client.save_kernel(request)
    print(json.dumps(response.to_dict(), indent=2))
    response_ref = str(getattr(response, "ref", "") or "")
    actual_slug = response_ref.rsplit("/", 1)[-1] if "/" in response_ref else args.slug
    final_status = _poll_kernel(client, resolved_username, actual_slug, args.timeout_seconds)

    output_dir = Path(args.output_dir)
    log_request = ApiListKernelSessionOutputRequest()
    log_request.user_name = resolved_username
    log_request.kernel_slug = actual_slug
    log_request.page_size = 100
    log_response = client.kernels.kernels_api_client.list_kernel_session_output(log_request)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "session.log").write_text(log_response.log, encoding="utf-8")
    _download_outputs(resolved_username, actual_slug, response.version_number, output_dir)
    print(f"kernel_url={response.url}")
    print(f"status={final_status.name}")
    print(f"output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
