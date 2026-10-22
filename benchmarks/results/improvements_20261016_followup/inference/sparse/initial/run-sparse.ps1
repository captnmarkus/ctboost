$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\apps\ctboost\.tmp\improvements-20261016\followup-native'
foreach ($label in @('dense-bins-1','sparse-bins-1','sparse-bins-2','dense-bins-2')) {
    $variant = if ($label.StartsWith('dense')) {'composed-cp310'} else {'sparse-cp310'}
    & C:/Users/captn/miniconda3/python.exe -I benchmarks/inference_real_followup.py --package "$studyRoot/$variant" --scout .tmp/inference-accuracy-20260908/accuracy-grouped-scout-v1 --output "$studyRoot/real" --label $label *> "$studyRoot/sparse-real-$label.log"
    if ($LASTEXITCODE -ne 0) {throw "Real sparse experiment failed: $label"}
    & C:/Users/captn/miniconda3/python.exe -I "$studyRoot/measure_synthetic.py" --package "$studyRoot/$variant" --fixtures .tmp/inference-accuracy-20260908/synthetic --output "$studyRoot/synthetic" --label $label *> "$studyRoot/sparse-synthetic-$label.log"
    if ($LASTEXITCODE -ne 0) {throw "Synthetic sparse experiment failed: $label"}
    Write-Output "$label complete"
}
