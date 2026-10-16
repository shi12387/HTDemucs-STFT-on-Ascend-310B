#!/usr/bin/env bash
# CANN environment for the standalone fixed-shape STFT project.
set -eo pipefail
HTSTFT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HTSTFT_TOOLKIT="${HTSTFT_CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
if [[ -f "$HTSTFT_TOOLKIT/../set_env.sh" ]]; then
    source "$HTSTFT_TOOLKIT/../set_env.sh"
elif [[ -f "$HTSTFT_TOOLKIT/set_env.sh" ]]; then
    source "$HTSTFT_TOOLKIT/set_env.sh"
fi
set -u
HTSTFT_CANN_INC="$HTSTFT_TOOLKIT/include"
HTSTFT_CANN_LIB="$HTSTFT_TOOLKIT/lib64"
if [[ ! -f "$HTSTFT_CANN_INC/acl/acl.h" ]]; then
    HTSTFT_CANN_INC="$HTSTFT_TOOLKIT/$(uname -m)-linux/include"
fi
if [[ ! -f "$HTSTFT_CANN_LIB/libascendcl.so" ]]; then
    HTSTFT_CANN_LIB="$HTSTFT_TOOLKIT/$(uname -m)-linux/lib64"
fi
