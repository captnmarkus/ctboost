$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\apps\ctboost\.tmp\improvements-20261016\followup-native'
$python = 'C:\apps\ctboost\.tmp\inference-accuracy-20260908\baseline-env\Scripts\python.exe'
$orders = @(
    @('ctboost','xgboost','lightgbm'),
    @('xgboost','lightgbm','ctboost'),
    @('lightgbm','ctboost','xgboost')
)
for ($index=0; $index -lt $orders.Count; $index++) {
    $label = "wave6-composed-order-$($index+1)"
    & $python -I benchmarks/fair_inference_scout.py --package-root "$studyRoot/composed-cp312" --dependencies .tmp/improvements-20261016/followup/deps --accuracy-results benchmarks/results/improvements_20261016_followup/accuracy_wave6_v2 --accuracy-artifacts .tmp/improvements-20261016/followup/accuracy_wave6_v2 --output "$studyRoot/fair/wave6-composed" --label $label --order $orders[$index] *> "$studyRoot/fair-$label.log"
    if ($LASTEXITCODE -ne 0) {throw "Fair latency diagnostic failed: $label"}
    Write-Output "$label complete"
}
