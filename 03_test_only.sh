#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")"
source ./cann_common.sh
PROJECT="$HTSTFT_ROOT/StftStandalone"
python3 check_project.py "$PROJECT"
API="$PROJECT/build_out/op_api"
export LD_LIBRARY_PATH="$API/lib:$HTSTFT_CANN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p bin results
DEVICE="${HTSTFT_DEVICE_ID:-0}"
[[ "$DEVICE" =~ ^[0-9]+$ ]] || { echo 'Invalid device ID' >&2; exit 1; }
g++ -std=c++11 -O2 -Wall -Wextra -ffp-contract=off runtime/main.cpp \
    -I"$API/include" -I"$HTSTFT_CANN_INC" -L"$HTSTFT_CANN_LIB" "$API/lib/libcust_opapi.so" \
    -lnnopbase -lascendcl -Wl,-rpath,"$API/lib" -Wl,-rpath,"$HTSTFT_CANN_LIB" \
    -Wl,-rpath-link,"$HTSTFT_CANN_LIB" -o bin/stft_test 2>&1 | tee app_build.log
python3 begin_run.py "$API/lib/libcust_opapi.so"
bin/stft_test "$DEVICE" "$HTSTFT_ROOT/test_data" "$HTSTFT_ROOT/results" 2>&1 | tee run.log
python3 validate.py 2>&1 | tee validate.log
