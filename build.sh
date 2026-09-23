/usr/local/cuda/bin/nvcc \
    -ccbin g++ \
    -std=c++20 \
    -DGPU_GRAPH_ENABLE_CUDA \
    -I./include \
    tests/simple_cuda.cu \
    -o simple_cuda
