#!/usr/bin/env bash

set -e

CUDA_BASE=/usr/local/cuda
# ALPAKA_BASE=/data/user/aalmarzo/alpaka_practice/alpaka
ALPAKA_BASE=/home/aalmarzouqi/Documents/cern_cms_collabration/2025_2026/unified_gpu_graphs/alpaka

NVCC="${CUDA_BASE}/bin/nvcc"
CXX=g++

SOURCE="${1:-tests/simple_cuda.cu}"
OUTPUT="${2:-simple_cuda}"

"${NVCC}" \
    -x cu \
    -ccbin "${CXX}" \
    -std=c++20 \
    -O2 \
    -g \
    -I./include \
    -I/usr/include/boost1.78 \
    -I"${ALPAKA_BASE}/include" \
    -DALPAKA_HAS_STD_ATOMIC_REF \
    -DALPAKA_ACC_GPU_CUDA_ENABLED \
    --expt-relaxed-constexpr \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -Xcompiler "-pthread" \
    "${SOURCE}" \
    -o "${OUTPUT}"

echo "Built: ${OUTPUT}"