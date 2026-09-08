import os
for key in ('CTBOOST_HIST_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import psutil
psutil.Process().cpu_affinity([6])
import argparse,cProfile,hashlib,json,pickle,pstats,statistics,sys,time
from pathlib import Path
import numpy as np
import ctboost
ROOT=Path(r'C:\apps\ctboost')
sys.path.insert(0,str(ROOT))
from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame

parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True)
parser.add_argument('--repeats',type=int,default=3)
args=parser.parse_args()
output=ROOT/'.tmp/inference-accuracy-20260908/hiva-profile'
output.mkdir(exist_ok=True)
source=ROOT/'.tmp/inference-accuracy-20260908/accuracy-grouped-scout-v1'
with (source/'fits/hiva_agnostic/ctboost_default/model.pkl').open('rb') as f:
    payload=pickle.load(f)
with (source/'data/hiva_agnostic/development.pkl').open('rb') as f:
    development=pickle.load(f)
raw=development['X'].iloc[np.arange(1000)%len(development['X'])].copy().reset_index(drop=True)
model=payload['model']
pipeline=model._feature_pipeline
def normalize():
    return normalize_tabarena_frame(raw,categorical_columns=payload['categorical_columns'])[0]
frame=normalize()
matrix,names=pipeline._extract_frame(frame)
if getattr(pipeline,'_numeric_only',False):
    matrix,names=pipeline._extract_frame(frame,allow_numeric=True)
transformed,categories,output_names=pipeline.transform_array(frame)
def build_pool():
    pool=ctboost.Pool(transformed,cat_features=categories,feature_names=output_names)
    pool._feature_pipeline=pipeline
    return pool
pool=build_pool()
state=pipeline.to_state()
def dtype_counts(frame):
    result={}
    for dtype in frame.dtypes:
        result[str(dtype)]=result.get(str(dtype),0)+1
    return result
def full():
    return model.predict_proba(normalize())
def timing(call):
    call()
    times=[]
    for _ in range(args.repeats):
        started=time.perf_counter(); call(); times.append((time.perf_counter()-started)*1000)
    return {'median_ms':statistics.median(times),'samples_ms':times}
report={'label':args.label,'native_sha256':hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest(),
        'pipeline_sha256':hashlib.sha256((Path(ctboost.__file__).parent/'feature_pipeline.py').read_bytes()).hexdigest(),
        'shape':list(raw.shape),'raw_dtypes':dtype_counts(raw),'normalized_dtypes':dtype_counts(frame),
        'raw_categorical_columns':len(payload['categorical_columns']),
        'numeric_only':getattr(pipeline,'_numeric_only',None),'matrix_dtype':str(matrix.dtype),
        'matrix_strides':list(matrix.strides),'transformed_shape':list(transformed.shape),
        'native_trees':int(model.get_booster()._handle.num_trees()),
        'state_counts':{k:len(state.get(k,[])) for k in ('numeric_indices','categorical_states','one_hot_states','combination_states','ctr_states','text_states','embedding_states')},
        'ctr_categories':sum(len(s['total_counts']) for s in state['ctr_states'])}
calls={'normalization':normalize,'extract_matrix':lambda:pipeline._extract_frame(frame),'native_transform':lambda:pipeline._native.transform_array(matrix,names),
       'pipeline_transform':lambda:pipeline.transform_array(frame),'state_refresh':pipeline._refresh_metadata,
       'build_pool':build_pool,'native_predict_prebuilt':lambda:model.get_booster()._handle.predict(pool._handle),
       'booster_predict_prebuilt':lambda:model.get_booster().predict(pool),'full_predict':full}
report['timings']={}
for key,call in calls.items():
    report['timings'][key]=timing(call)
    print(json.dumps({'label':args.label,'stage':key,**report['timings'][key]}),flush=True)
profile=cProfile.Profile()
profile.runcall(full)
profile.dump_stats(str(output/f'{args.label}.prof'))
with (output/f'{args.label}.txt').open('w') as f:
    pstats.Stats(profile,stream=f).sort_stats('cumtime').print_stats(30)
    pstats.Stats(profile,stream=f).sort_stats('tottime').print_stats(20)
prediction=full()
np.save(output/f'{args.label}.npy',prediction)
report['prediction_sha256']=hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest()
(output/f'{args.label}.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='timings'}),flush=True)
