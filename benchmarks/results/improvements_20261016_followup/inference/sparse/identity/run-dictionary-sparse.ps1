$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\apps\ctboost\.tmp\improvements-20261016\followup-native'
foreach ($label in @('layout-base-1','dictionary-1','identity-1','identity-2','dictionary-2','layout-base-2')) {
    $variant = if ($label.StartsWith('layout')) {'composed-cp310'} elseif ($label.StartsWith('dictionary')) {'category-codes-cp310'} else {'sparse-identity-cp310'}
    & C:/Users/captn/miniconda3/python.exe -I benchmarks/inference_real_followup.py --package "$studyRoot/$variant" --scout .tmp/inference-accuracy-20260908/accuracy-grouped-scout-v1 --output "$studyRoot/real" --label $label *> "$studyRoot/layout-real-$label.log"
    if ($LASTEXITCODE -ne 0) {throw "Real layout experiment failed: $label"}
    & C:/Users/captn/miniconda3/python.exe -I "$studyRoot/measure_synthetic.py" --package "$studyRoot/$variant" --fixtures .tmp/inference-accuracy-20260908/synthetic --output "$studyRoot/synthetic" --label $label *> "$studyRoot/layout-synthetic-$label.log"
    if ($LASTEXITCODE -ne 0) {throw "Synthetic layout experiment failed: $label"}
    Write-Output "$label complete"
}
