import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
CASES=('tones_warmup','impulse','tones','noise','zero','tones_restore')
INPUT_SHAPE=(2,351232);OUTPUT_SHAPE=(1,4,2048,336)

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def info(path):
    p=Path(path);return {'bytes':p.stat().st_size,'sha256':sha(p)}
def checked(path,m):
    p=Path(path);require(p.is_file() and p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],'Missing/changed file: '+str(p))
def load(path,shape):
    x=np.fromfile(path,dtype='<f4');require(x.size==int(np.prod(shape)) and np.isfinite(x).all(),'Invalid FP32 tensor: '+str(path));return x.reshape(shape)
def sources():
    paths=list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))+[ROOT/'baseline_sources.json']
    for d in ('op_host','op_kernel','runtime'):paths.extend((ROOT/d).glob('*'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()}
def baseline():
    d=json.loads((ROOT/'baseline_sources.json').read_text())
    require(len(d['files'])==12,'Expected twelve baseline files')
    for rel,digest in d['files'].items():require(sha(ROOT/rel)==digest,'Core source differs from baseline: '+rel)
def inputs():
    m=json.loads((ROOT/'test_data/manifest.json').read_text());require(m['cases']==list(CASES),'Unexpected test sequence')
    require(m['input_shape']==list(INPUT_SHAPE) and m['output_shape']==list(OUTPUT_SHAPE),'Changed fixed shape')
    for name,v in m['files'].items():checked(ROOT/'test_data'/name,v)
    for rel,digest in m['sources'].items():require(sha(ROOT/rel)==digest,'Source changed after test preparation: '+rel)
    return m
