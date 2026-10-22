$ErrorActionPreference = 'Stop'
$studyRoot = 'C:\apps\ctboost\.tmp\improvements-20261016\followup-native'
foreach ($label in @('codes-base-1','codes-native-1','codes-native-2','codes-base-2')) {
    $variant = if ($label.StartsWith('codes-base')) {'composed-cp310'} else {'codes-native-cp310'}
    & C:/Users/captn/miniconda3/python.exe -I benchmarks/inference_real_followup.py --package "$studyRoot/$variant" --scout .tmp/inference-accuracy-20260908/accuracy-grouped-scout-v1 --output "$studyRoot/real" --label $label *> "$studyRoot/real-$label.log"
    if ($LASTEXITCODE -ne 0) {throw "Native categorical-code experiment failed: $label"}
    Write-Output "$label complete"
}
