import json,sys
from pathlib import Path
from support import ROOT,inputs,info,sha
m=inputs();out=ROOT/'results';out.mkdir(exist_ok=True)
for name in ('stft_pass.json','validation.json','run_report.json'):(out/name).unlink(missing_ok=True)
p=Path(sys.argv[1]).resolve();(out/'run_context.json').write_text(json.dumps({'library':{'path':str(p),**info(p)},'test_manifest_sha256':sha(ROOT/'test_data/manifest.json')},indent=2)+'\n')
