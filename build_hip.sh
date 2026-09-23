#!/usr/bin/env bash

set -euo pipefail

ALPAKA_BASE=./alpaka
HIPCC=hipcc

SOURCE="${1:-tests/create_destory_graph_hip.cpp}"
OUTPUT="${2:-create_destory_graph_hip.out}"

"${HIPCC}" \
    -std=c++20 \
    -O2 \
    -g \
    -I./include \
    -I"${ALPAKA_BASE}/include" \
    -DALPAKA_HAS_STD_ATOMIC_REF \
    -DALPAKA_ACC_GPU_HIP_BACKEND \
    -pthread \
    "${SOURCE}" \
    -o "${OUTPUT}"

echo "Built ${OUTPUT}"