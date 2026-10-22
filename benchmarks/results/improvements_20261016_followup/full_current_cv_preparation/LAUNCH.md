This is an inactive launch recipe. Do not execute until root confirms that the completed, numerically successful 17b primary gate failed, or that its linked 569 mean gate failed, and that the machine is quiet for the RAM snapshot. The immutable accuracy runtime is c5; the final inference runtime 04ad is not used here. No real plan, folds, fitting, or scores were produced during preparation.

The accuracy agent owns eventual execution. Preserve these new sources and all parent dependencies after preparation. Any numerical, memory, wall-time, or cleanup failure is terminal for this protocol; do not alter limits or fall back to parent scores/models.

From `C:\apps\ctboost`, use these explicit PowerShell arguments:

```powershell
$cvPython = 'C:\apps\ctboost\.tmp\inference-accuracy-20260908\baseline-env\Scripts\python.exe'
$cvMain = 'C:\apps\ctboost\benchmarks\fair_accuracy_current_round_cv.py'
$cvSeed = 'C:\apps\ctboost\benchmarks\fair_accuracy_current_round_cv_seed_check.py'
$cvOuter = 'C:\apps\ctboost\benchmarks\fair_accuracy_current_round_cv_outer_test.py'
$cvCommon = @(
  '--output', 'C:\apps\ctboost\benchmarks\results\improvements_20261016_followup\accuracy_wave18',
  '--artifacts', 'C:\apps\ctboost\.tmp\improvements-20261016\followup\accuracy_wave18',
  '--package-root', 'C:\apps\ctboost\.tmp\improvements-20261016\followup\runtime',
  '--dependencies', 'C:\apps\ctboost\.tmp\improvements-20261016\followup\deps',
  '--previous', 'C:\apps\ctboost\benchmarks\results\improvements_20261016_followup\accuracy_wave17b'
)
$cvResources = @(
  '--cpu', '4', '--maximum-named-array-bytes', '6442450944',
  '--maximum-job-rss-bytes', '10737418240', '--maximum-job-seconds', '1200',
  '--headroom-reserve-bytes', '4294967296', '--headroom-wait-seconds', '300',
  '--headroom-poll-seconds', '5'
)
& $cvPython -I $cvMain prepare @cvCommon @cvResources
```

Only for the linked-569 failure branch, append `--failed-seed C:\apps\ctboost\benchmarks\results\improvements_20261016_followup\accuracy_wave17b_seed569` to `prepare`. The admission code verifies that directory's exact source/runtime/plan/model/development chain. Do not supply it after an initial primary failure. Both branches use the primary 47 models and controls.

Before dispatch, inspect and record the plan hash, all 216 prospective fold/full-union bounds, and the quiet-memory receipt. The fixed contract is one worker, one thread, CPU4, 6 GiB named arrays, 10 GiB process-tree RSS, 1,200 seconds per fit, and at least 14 GiB available RAM before dispatch. Headroom waits are bounded at 300 seconds with five-second polls. The preparation snapshot precedes loading fitting frames. Resource values cannot change after freeze.

Run stages sequentially and inspect each receipt before proceeding:

```powershell
& $cvPython -I $cvMain run_folds @cvCommon
# Require fold_completion.json: all_fits_successful=true, 540 fully_verified_jobs.
& $cvPython -I $cvMain select @cvCommon
# Preserve selected_cv.json hash: all36 choices, each re-derived from five fold errors.
& $cvPython -I $cvMain run_union @cvCommon
# Require frozen_selections.json: all_fits_successful=true, 36 fully_verified_jobs.
& $cvPython -I $cvMain development @cvCommon
```

These are 540 planned fold jobs and 36 mandatory union jobs. Each receipt distinguishes a started native fit, a completed native return, and a fully verified correction/reload/persistence job. A failed correction is not a successful job. Fold split seed18018 and learner seed47 are fixed. Duplicate rounded profiles remain distinct; selection is the mean of five primary errors, tied by fewer requested rounds then profile index. The `pause_fits` marker in the artifact directory pauses between processes without changing resources.

Only after the primary quality gate passes, use fresh directories `accuracy_wave18_seed677` in both report and artifact roots. Run the seed companion with stages `prepare`, `run`, `development`; retain the same package/dependency arguments and set `--previous` to the wave18 report directory. No resource arguments or selection are accepted by the seed companion: it inherits the exact frozen resources, policies, authorized fitting rows, counts, corrections, and temperature and changes only the learner seed to677. All36 fresh fits must succeed. Aggregate 47/677 error means within the same12 tasks.

Only after both quality gates pass, use fresh directories `accuracy_wave18_outer` in the report/artifact roots. Run the outer companion `prepare` with the same runtime/dependencies and `--previous` pointing to the seed677 report. This freezes72 already-trained models and source/split hashes without opening test values. Root must coordinate the one-shot `assess` stage; it performs no fits or model replacement. Assessment reads the full source parquet, including rows from the original confirmation role, but never opens that role file or selects on/separately scores that role. Any assessment-start marker closes future adaptive-wave admission.

The final serving/timing policy remains primary seed47 of the first policy passing both gates. Training and all quality work finish before an exclusive inference timing window. Cross-validation errors and repeatedly used development scores are model-selection evidence, not unbiased external estimates or a universal ranking.
