#!/usr/bin/env python3
"""Check every packed STFT element; no model normalization or model inference."""
import csv,json,math
import numpy as np
from support import ROOT,CASES,INPUT_SHAPE,OUTPUT_SHAPE,require,load,inputs,sha,checked,info

def metrics(a,b):
    e=a.astype(np.float64)-b.astype(np.float64);sq=float(np.sum(e*e));energy=float(np.sum(b.astype(np.float64)**2));rel=math.sqrt(sq/energy) if energy else (0 if sq==0 else float('inf'))
    return {'checked_elements':int(a.size),'bitwise_difference_count':int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),
            'mismatch_count':int(np.count_nonzero(np.abs(e)>2e-5+2e-5*np.abs(b.astype(np.float64)))),
            'max_abs_error':float(np.max(np.abs(e))),'rmse':math.sqrt(sq/a.size),'relative_l2_error':rel,'atol':2e-5,'rtol':2e-5}

def main(board=True):
    out=ROOT/'results';marker=out/'stft_pass.json';marker.unlink(missing_ok=True);m=inputs()
    context=json.loads((out/'run_context.json').read_text());require(context['test_manifest_sha256']==sha(ROOT/'test_data/manifest.json'),'Test data changed during the run');checked(context['library']['path'],context['library'])
    r=json.loads((out/'run_report.json').read_text());require(r['status']=='RUN_COMPLETE_REQUIRES_NUMERICAL_VALIDATION','ACL test did not complete')
    require([v['name'] for v in r['runs']]==list(CASES),'Wrong case order')
    require(r['constant_uploads']==3 and r['constant_bytes']==67649536 and r['workspace_bytes']==2097152,'Unexpected cached constants/workspace')
    checks={};outputs={}
    for i,(name,row) in enumerate(zip(CASES,r['runs'])):
        require(row['warmup']==(i==0) and row['kernel_launches']==7571 and row['syncs']==46 and row['intermediate_application_copies']==0,'Invalid live-chain accounting: '+name)
        require(row['input_bytes']==2809856 and row['output_bytes']==11010048,'Changed transfer sizes')
        require(all(math.isfinite(v) and v>=0 for k,v in row.items() if k.endswith('_ms')),'Invalid timing')
        a=load(out/(name+'_npu_f32.bin'),OUTPUT_SHAPE);b=load(ROOT/'test_data'/(name+'_reference_f32.bin'),OUTPUT_SHAPE);v=metrics(a,b)
        require(v['mismatch_count']==0 and v['relative_l2_error']<=2e-5,'STFT error exceeds tolerance: '+name+' '+json.dumps(v));checks[name]=v;outputs[name+'_npu_f32.bin']=info(out/(name+'_npu_f32.bin'))
        if name=='zero':require(np.count_nonzero(a)==0,'Zero input retained nonzero stale output')
        print('[CHECK]',name,json.dumps(v))
    first=load(out/'tones_npu_f32.bin',OUTPUT_SHAPE);restored=load(out/'tones_restore_npu_f32.bin',OUTPUT_SHAPE)
    require(np.array_equal(first.view(np.uint32),restored.view(np.uint32)),'Same-address tones/zero/tones output changed')
    with (out/'timing_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r['runs'][0]));w.writeheader();w.writerows(r['runs'])
    value={'status':'PASS_STANDALONE_STFT_ALIGNMENT' if board else 'PASS_LOCAL_STFT_ORCHESTRATION_SIMULATION','reference_backend':m['reference_backend'],'checks':checks,'restore_bitwise_mismatch_count':0,
           'input_shape':list(INPUT_SHAPE),'output_shape':list(OUTPUT_SHAPE),'library':context['library'],'outputs':outputs,
           'test_manifest_sha256':sha(ROOT/'test_data/manifest.json'),'run_report_sha256':sha(out/'run_report.json'),
           'scope':'Fixed-shape pre-padded stereo STFT, normalized=True, cropped and packed CaC; no waveform padding, model mean/std normalization, OMs or iSTFT.',
           'timing_scope':'Test cases use different signals. steady_clock launch+sync wall times, not device-event kernel times. Initialization/constants, sentinel resets, file I/O and CPU verification are excluded from segment_wall_ms.'}
    (out/'validation.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');marker.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(value['status']);return value
if __name__=='__main__':main()
