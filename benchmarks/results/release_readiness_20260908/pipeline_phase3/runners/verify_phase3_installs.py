"""Bind completed pipeline tests to all bytes in the finalized wheel payload."""
import hashlib
import json
import zipfile
from pathlib import Path

root = Path(r'C:\apps\ctboost/.tmp/release-readiness-20260908')
wheel_path = root / 'dist-portable/ctboost-0.1.61-cp312-cp312-win_amd64.whl'
wheel_sha = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
assert wheel_sha == 'b9fb7b598395a64380ee2f5a2395269ef567169edd151916790087dd71177f94'
environments = []
with zipfile.ZipFile(wheel_path) as wheel:
    members = [name for name in wheel.namelist() if not name.endswith('/')]
    record = [name for name in members if name.endswith('.dist-info/RECORD')]
    assert len(record) == 1
    checked = [name for name in members if name not in record]
    hashes = {name: hashlib.sha256(wheel.read(name)).hexdigest() for name in checked}
    for env_name in ('pipeline-py312-np1', 'pipeline-py312-np2'):
        site = root / env_name / 'Lib/site-packages'
        for name, digest in hashes.items():
            installed = site / name
            assert installed.is_file() and hashlib.sha256(installed.read_bytes()).hexdigest() == digest, (env_name, name)
        for package in ('ctboost', 'benchmarks'):
            installed_paths = {file.relative_to(site).as_posix() for file in (site / package).rglob('*')
                               if file.is_file() and '__pycache__' not in file.parts}
            expected_paths = {name for name in checked if name.startswith(package + '/')}
            assert installed_paths == expected_paths, (env_name, 'unexpected package files', installed_paths ^ expected_paths)
        environments.append({'environment': env_name, 'checked_wheel_members': len(checked),
                             'all_payload_bytes_equal': True, 'no_extra_package_files': True})
report = {
    'wheel_sha256': wheel_sha, 'wheel_member_count': len(members),
    'checked_member_sha256': hashes, 'environments': environments,
    'excluded_metadata': record,
    'exclusion_reason': 'The installer rewrites RECORD to include generated installation metadata; all other wheel members match.',
    'identity_correction': {
        'early_hash_while_archive_was_still_being_written': 'b3c83491c777ed79b7f93418c54dfccb96f2fbcf4da0402320f197b153e8026f',
        'completed_wheel_sha256': wheel_sha,
        'tests_repeated': False,
        'reason': 'All tested installed code and package data match the completed ZIP byte-for-byte.',
    },
}
(root / 'phase3-installed-wheel-identity.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps({'wheel_sha256': wheel_sha, 'environments': environments}))
