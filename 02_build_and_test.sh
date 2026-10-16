#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")"
source ./cann_common.sh
PROJECT="$HTSTFT_ROOT/StftStandalone"
[[ -d "$PROJECT" ]] || { echo 'Run 01_prepare.sh first' >&2; exit 1; }
(cd "$PROJECT" && bash ./build.sh) 2>&1 | tee build.log
bash ./03_test_only.sh
