"""Closed headroom failure audit and prospective stricter-RSS recommendation."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'accuracy_wave18'
HASHES={}


def sha(path):
    path=Path(path);value=hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)]=value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    plan=read(OUT/'plan.json');done=read(OUT/'fold_completion.json');summary=read(OUT/'summary.json')
    prepared=read(HERE/'wave18_preparation_independent_review.json')
    assert done['plan_sha256']==prepared['plan_sha256']==sha(OUT/'plan.json')
    assert done['all_fits_successful'] is False and summary['status']=='terminal_fold_failure'
    paths=[Path(p) for p in done['records']]
    assert len(paths)==len(set(paths))==540 and set(paths)==set((OUT/'fold_jobs').glob('*.json'))
    records=[read(p) for p in paths]
    keys={(r['dataset'],r['library'],r['profile_index'],r['fold']) for r in records}
    assert len(keys)==540
    expected={(j['dataset'],j['library'],p,f) for j in plan['jobs'] for p in range(3) for f in range(5)}
    assert keys==expected
    assert all(r['status']=='ok' and r['native_fit_started'] is True and r['native_fit_completed'] is True for r in records[:58])
    failed=records[58]
    assert failed['failure']=='insufficient_system_headroom' and not failed.get('native_fit_started',False)
    assert all(r['failure']=='not_started_after_terminal_failure' and not r.get('native_fit_started',False) and not r.get('native_fit_completed',False) for r in records[59:])
    assert summary['fold_completion_counts']==done['completion_counts']
    assert done['completion_counts']['native_fits_completed']==done['completion_counts']['native_fits_started']==done['completion_counts']['fully_verified_jobs']==58
    headroom=failed['headroom']
    assert summary['headroom_failures']==[headroom] and not headroom['sufficient']
    required=plan['resources']['maximum_job_rss_bytes']+plan['resources']['headroom_reserve_bytes']
    assert headroom['required_available_bytes']==required==14*2**30
    samples=headroom['samples'];assert len(samples)==61 and samples[-1]['elapsed_seconds']==300
    assert all(s['available_bytes']<required for s in samples)
    resources=[read(p) for p in sorted((OUT/'fold_job_logs').glob('*-resource.json'))]
    assert len(resources)==58 and all(r['exit_code']==0 and r['resource_failure'] is None and r['cleanup_complete'] is True for r in resources)
    assert summary['total_quality_tree_fits_through_wave18']==2472+58==2530
    assert summary['not_started_after_terminal_failure']==481 and summary['resource_failures']==[]
    assert not summary['development_scored'] and summary['development_target_met'] is None
    assert all(not (OUT/name).exists() for name in ('selected_cv.json','frozen_selections.json','development.json'))
    for name,expected_sha in summary['source_sha256'].items():assert sha(OUT/name)==expected_sha
    for path,expected_sha in plan['dependencies'].items():
        if Path(path).suffix=='.py':assert sha(path)==expected_sha
    assert prepared['maximum_active_named_bound_bytes']==4515168888<6*2**30
    receipt={
        'date':'2026-10-16','status':'terminal_headroom_failure_verified_and_stricter_retry_feasible_to_attempt',
        'plan_sha256':sha(OUT/'plan.json'),'fold_completion_sha256':sha(OUT/'fold_completion.json'),
        'summary_sha256':sha(OUT/'summary.json'),'preparation_audit_sha256':sha(HERE/'wave18_preparation_independent_review.json'),
        'completed_model_training_runs':58,'all_completed_jobs_verified':True,'not_started_total':482,
        'trigger':'insufficient_system_headroom','wait_seconds':300,'headroom_samples':61,
        'required_available_bytes_original':required,'maximum_available_during_wait':max(s['available_bytes'] for s in samples),
        'final_available_during_wait':samples[-1]['available_bytes'],'selection_union_development_run':False,
        'cumulative_benchmark_model_fits':2530,'diagnostic_replay_fits_separate':8,
        'prospective_retry':{'output_identity':'accuracy_wave18b','generic_plan_wave':18,
            'maximum_job_rss_bytes':8*2**30,'maximum_named_array_bytes':6*2**30,'headroom_reserve_bytes':4*2**30,
            'required_available_bytes':12*2**30,'maximum_job_seconds':1200,'workers':1,'threads_per_fit':1,'cpu':4,
            'headroom_wait_seconds':300,'headroom_poll_seconds':5,'fresh_fold_fits':540,'fresh_union_fits':36,
            'old_fit_reuse':0,'method_source_or_quality_gate_change':False,'new_amendment_and_plan_required':True},
        'feasibility':'All216 previously verified named bounds remain below6GiB; worst4,515,168,888bytes.8GiB RSS is a stricter process cap and12GiB dispatch minimum retains4GiB host reserve.',
        'limits':'Named allocations omit frames/models/CTR/allocator and transient overhead. Prior2,434,367,488byte peak does not predict double-count CV memory by a fixed multiplier. No guarantee of convergence or8GiB compliance; unchanged terminal RSS/time/headroom/solver gates remain authoritative. This review does not activate the retry.',
        'access':'Closed JSON/source hashes only; no raw data/model/prediction decode, fits, tests or scoring.',
        'audit_source_sha256':sha(__file__),'verified_file_sha256':HASHES}
    target=HERE/'wave18_terminal_resource_retry_review.json';assert not target.exists()
    temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');temp.replace(target)
    print(json.dumps({'status':receipt['status'],'receipt':str(target),'receipt_sha256':sha(target)}))


if __name__=='__main__':main()
