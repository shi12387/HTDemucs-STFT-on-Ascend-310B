"""Extend the board-generated SIMPLE project to four registered operators."""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OPS=[('HtStftFrameWindow', 'ht_stft_frame_window'), ('HtStftSplit', 'ht_stft_split'), ('HtDftMatmul', 'ht_dft_matmul'), ('HtStftMergePack', 'ht_stft_merge_pack')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("project",type=Path);args=parser.parse_args()
    project=args.project.resolve()
    for name,stem in OPS:
        for rel in ("op_host/"+stem+".cpp","op_host/"+stem+"_tiling.h","op_kernel/"+stem+".cpp"):
            shutil.copy2(ROOT/rel,project/rel)
    host_path=project/"op_host/CMakeLists.txt"
    text=host_path.read_text()
    aux=re.search(r"aux_source_directory\s*\(\s*[^\s)]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\)",text,re.I)
    glob=None
    if aux is None:
        for match in re.finditer(r"file\s*\(\s*GLOB(?:_RECURSE)?\s+([A-Za-z_][A-Za-z0-9_]*)\s+[^)]*\)",text,re.I):
            body=match.group(0)
            if ("CMAKE_CURRENT_SOURCE_DIR" in body or "ht_dft_matmul.cpp" in body) and ".cpp" in body:
                glob=match;break
    match=aux or glob
    if match is None:
        raise RuntimeError("Unrecognized generated op_host source discovery; share op_host/CMakeLists.txt before changing build workflow")
    variable=match.group(1)
    declaration="\n# Standalone STFT: explicit multi-op host source list.\nset("+variable+"\n"
    declaration+="".join('    "${CMAKE_CURRENT_SOURCE_DIR}/'+stem+'.cpp"\n' for _,stem in OPS)+")\n"
    host_path.write_text(text[:match.end()]+declaration+text[match.end():])
    kernel_path=project/"op_kernel/CMakeLists.txt"
    text=kernel_path.read_text()
    if "add_kernel_compile" not in text:
        raise RuntimeError("Expected CANN8 SIMPLE add_kernel_compile; share op_kernel/CMakeLists.txt")
    additions=[]
    for name,stem in OPS:
        if not re.search(r'add_kernel_compile\s*\(\s*"?'+re.escape(name)+r'"?(?:\s|\))',text):
            additions.append('add_kernel_compile('+name+' "${CMAKE_CURRENT_SOURCE_DIR}/'+stem+'.cpp")')
    kernel_path.write_text(text+"\n# Standalone STFT additional kernels.\n"+"\n".join(additions)+"\n")
    func_path=project/"cmake/func.cmake"
    func_text=func_path.read_text()
    anchor='  message(STATUS "Opbuild generating sources - done")'
    if func_text.count(anchor)!=1:raise RuntimeError("Unexpected CANN opbuild function layout")
    codegen=project/"cmake/util/htstft_preload_codegen.py"
    shutil.copy2(ROOT/"resource_preload_codegen.py",codegen)
    block='\n  # Standalone STFT: board-proven explicit resource preload codegen.\n  execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE}\n                  ${CMAKE_SOURCE_DIR}/cmake/util/htstft_preload_codegen.py ${OPBUILD_OUT_DIR}\n                  RESULT_VARIABLE HTSTFT_PRELOAD_RESULT\n                  OUTPUT_VARIABLE HTSTFT_PRELOAD_INFO\n                  ERROR_VARIABLE HTSTFT_PRELOAD_ERROR)\n  if(NOT HTSTFT_PRELOAD_RESULT EQUAL 0)\n    message(FATAL_ERROR "Preload codegen failed: ${HTSTFT_PRELOAD_INFO} ${HTSTFT_PRELOAD_ERROR}")\n  endif()\n  message(STATUS "${HTSTFT_PRELOAD_INFO}")\n'
    func_path.write_text(func_text.replace(anchor,anchor+block))
    shutil.copytree(ROOT/"runtime",project/"runtime",dirs_exist_ok=True)
    print("[UNIFIED BUILD] "+json.dumps({"host_source_variable":variable,"ops":[x[0] for x in OPS],
                                       "added_kernel_entries":len(additions)}),flush=True)


if __name__=="__main__":main()
