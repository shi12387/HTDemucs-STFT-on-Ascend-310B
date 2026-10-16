#!/usr/bin/env bash
set -eo pipefail
umask 022
cd -- "$(dirname -- "$0")"
source ./cann_common.sh
PROJECT="$HTSTFT_ROOT/StftStandalone"
[[ ! "$HTSTFT_ROOT" =~ [^a-zA-Z0-9_./-] ]] || { echo 'Use an ASCII path without spaces, e.g. ~/ascend_work/ascend310b_htdemucs_stft' >&2; exit 1; }
[[ ! -e "$PROJECT" ]] || { echo 'Project exists: use 02_build_and_test.sh. Regenerate test data separately with python3 prepare_tests.py.' >&2; exit 1; }
python3 prepare_tests.py "$@"
chmod go-w "$HTSTFT_ROOT" "$HTSTFT_ROOT/anchor_dft_matmul.json"
msopgen gen -i "$HTSTFT_ROOT/anchor_dft_matmul.json" -f aclnn -c ai_core-ascend310b -lan cpp -out "$PROJECT" 2>&1 | tee prepare.log
python3 configure_project.py "$PROJECT" 2>&1 | tee -a prepare.log
echo 'PREPARED: next bash 02_build_and_test.sh'
