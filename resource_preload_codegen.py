#!/usr/bin/env python3
import re
import sys
from pathlib import Path

OPS=[('HtStftFrameWindow', 'ht_stft_frame_window'), ('HtStftSplit', 'ht_stft_split'), ('HtDftMatmul', 'ht_dft_matmul'), ('HtStftMergePack', 'ht_stft_merge_pack')]
MARKER="// Standalone STFT shared lazy resource-id getter v1"


def transform(text,name):
    getter="HtStftStandaloneResourceId_"+name
    export="HtStftStandalonePreload_"+name
    if MARKER in text:
        if getter not in text or export not in text:
            raise ValueError("Incomplete previous patch: "+name)
        return text
    pattern=(r'\bstatic\s+uint32_t\s+'+re.escape(name)+r'OpTypeId\s*=\s*'
             r'op::GenOpTypeId\(\s*"'+re.escape(name)+r'"\s*,\s*'+re.escape(name)+r'_RESOURCES\s*\)\s*;')
    matches=list(re.finditer(pattern,text))
    if len(matches)!=1:raise ValueError("Unexpected generated registration statement: "+name)
    text=re.sub(pattern,"const uint32_t "+name+"OpTypeId = "+getter+"();",text)
    insertion=text.find("namespace {")
    if insertion<0 or 'extern uint32_t GenOpTypeId(' not in text[:insertion]:
        raise ValueError("Unexpected generated wrapper layout: "+name)
    prefix=(MARKER+'\n#ifdef ACLNN_WITH_BINARY\nnamespace {\n'
            'uint32_t '+getter+'() {\n'
            '    static uint32_t id = op::GenOpTypeId("'+name+'", '+name+'_RESOURCES);\n'
            '    return id;\n}\n}\n'
            'extern "C" __attribute__((visibility("default"))) uint32_t '+export+'() {\n'
            '    return '+getter+'();\n}\n#endif\n\n')
    return text[:insertion]+prefix+text[insertion:]


def main():
    root=Path(sys.argv[1]).resolve();changes=[]
    for name,stem in OPS:
        path=root/("aclnn_"+stem+".cpp")
        original=path.read_text();changed=transform(original,name)
        changes.append((path,changed))
    for path,changed in changes:path.write_text(changed)
    print("[RESOURCE PRELOAD CODEGEN] four wrappers share their existing resource-id getters",flush=True)


if __name__=="__main__":main()
