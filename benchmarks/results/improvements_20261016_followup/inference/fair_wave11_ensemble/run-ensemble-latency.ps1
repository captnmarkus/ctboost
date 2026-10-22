$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\apps\ctboost\.tmp\improvements-20261016\followup-native'
$planPath = "$studyRoot/fair/wave11-ensemble/latency-plan.json"
$timingPlan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
foreach ($entry in $timingPlan.frozen_files.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
        throw "Frozen timing dependency changed: $($entry.Name)"
    }
}
foreach ($pass in $timingPlan.passes) {
    $label = $pass.label
    & .tmp/inference-accuracy-20260908/baseline-env/Scripts/python.exe -I benchmarks/fair_inference_ensemble_scout.py --package-root "$studyRoot/final-codes-cp312" --dependencies .tmp/improvements-20261016/followup/deps --accuracy-results benchmarks/results/improvements_20261016_followup/accuracy_wave11 --accuracy-artifacts .tmp/improvements-20261016/followup/accuracy_wave11 --output "$studyRoot/fair/wave11-ensemble" --label $label --order $pass.order --reference "$studyRoot/fair/wave11-ensemble/wave11-final-check-v2.json" *> "$studyRoot/ensemble-latency-$label.log"
    if ($LASTEXITCODE -ne 0) { throw "Frozen ensemble latency failed: $label" }
    Write-Output "$label complete"
}
