## Build and run

### 1. Clone repository

```bash
git clone https://github.com/hamani115/alpaka_gpu_graph.git
cd alpaka_gpu_graph
```

### 2. Clone alpaka

Clone alpaka inside the repository:

```bash
git clone https://github.com/alpaka-group/alpaka.git
```

The directory structure should look like:

```text
alpaka_gpu_graph/
├── alpaka/
├── include/
├── tests/
└── build_cuda.sh
```

### 3. Set the alpaka path

Open `build_cuda.sh` and set:

```bash
ALPAKA_BASE=./alpaka
```

For example, the beginning of the script should contain:

```bash
CUDA_BASE=/usr/local/cuda
ALPAKA_BASE=./alpaka
```

### 4. Build a CUDA test

Run the CUDA build script from repository root:

```bash
bash build_cuda.sh <cuda-file> <output-name>
```

For example:

```bash
bash build_cuda.sh tests/simple_cuda.cu simple_cuda.out
```

### 5. Run the Executable

```bash
./simple_cuda.out
```
