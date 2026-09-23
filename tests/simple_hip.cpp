#include <gpu_graph/Graph.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <hip/hip_runtime.h>

__global__ void addKernel(double *arrayA, const int *arrayB, size_t size)
{
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x < size)
    {
        arrayA[x] += arrayB[x];
    }
}

__global__ void multiplyKernel(double *arrayA, size_t size)
{
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x < size)
    {
        arrayA[x] *= 2.0;
    }
}

__global__ void assignKernel(int *arrayB, size_t size)
{
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x < size)
    {
        arrayB[x] = 3;
    }
}

int main()
{
    using Backend = gpu_graph::HipBackend;

    auto stream = gpu_graph::createStream<Backend>();
    auto graph = gpu_graph::createGraph<Backend>();

    unsigned int const iters = 10;
    size_t n{4};

    double *d_arrayA{};
    int *d_arrayB{};

    hipMallocManaged(&d_arrayA, n * sizeof(double));
    hipMallocManaged(&d_arrayB, n * sizeof(int));

    for (size_t i = 0; i < n; ++i)
    {
        d_arrayA[i] = static_cast<double>(i);
        d_arrayB[i] = 0;
    }

    unsigned int const threadsPerBlock = 256u;

    unsigned int const blocksPerGrid = static_cast<unsigned int>(
        (n + tassignKernelhreadsPerBlock - 1u) / threadsPerBlock);

    float *d_input{};
    float *d_output{};
    std::size_t n{};

    // Node 1: assignKernel

    void *assignArgs[] = {
        &d_arrayB,
        &n,
    };

    gpu_graph::KernelNodeConfig const assignConfig{
        reinterpret_cast<void *>(assignKernel),
        {blocksPerGrid, 1u, 1u},
        {threadsPerBlock, 1u, 1u},
        assignArgs,
    };

    auto const assignNode = gpu_graph::addKernelNode<Backend>(
        graph,
        assignConfig);

    // Node 2: addKernel -> Depends on assignKernel

    void *addArgs[] = {
        &d_arrayA,
        &d_arrayB,
        &n,
    };

    gpu_graph::KernelNodeConfig const addConfig{
        reinterpret_cast<void *>(addKernel),
        {blocksPerGrid, 1u, 1u},
        {threadsPerBlock, 1u, 1u},
        addArgs,
    };

    std::array<gpu_graph::Node<Backend>, 1u> const addDependencies{
        assignNode,
    };

    auto const assignNode = gpu_graph::addKernelNode<Backend>(
        graph,
        addDependencies,
        addConfig);

    // Node 3: multiplyKernel -> depends on addKernel

    void *multiplyArgs[] = {
        &d_arrayA,
        &n,
    };

    gpu_graph::KernelNodeConfig const multiplyConfig{
        reinterpret_cast<void *>(multiplyKernel),
        {blocksPerGrid, 1u, 1u},
        {threadsPerBlock, 1u, 1u},
        multiplyArgs,
    };

    std::array<gpu_graph::Node<Backend>, 1u> const multiplyDependencies{
        addNode,
    };

    auto const multiplyNode = gpu_graph::addKernelNode<Backend>(
        graph,
        multiplyDependencies,
        multiplyConfig);

    auto executable = gpu_graph::instantiate<Backend>(graph);

    // ============================================================

    // Launch graph multiple times

    for (std::size_t iteration = 0u; iteration < iters; ++iteration)
    {
        gpu_graph::launch<Backend>(executable, stream);
    }

    gpu_graph::synchronize<Backend>(stream);

    // ============================================================

    // Print results

    std::cout << "arrayA:" << std::endl;

    for (size_t i = 0; i < n; ++i)
    {
        std::cout << "arrayA[" << i << "] = " << d_arrayA[i] << std::endl;
    }

    std::cout << "\narrayB:" << std::endl;

    for (size_t i = 0; i < n; ++i)
    {
        std::cout << "arrayB[" << i << "] = " << d_arrayB[i] << std::endl;
    }

    // ============================================================

    // Cleanup

    gpu_graph::destroyExecutable<Backend>(executable);
    gpu_graph::destroyGraph<Backend>(graph);
    gpu_graph::destroyStream<Backend>(stream);

    hipFree(d_arrayA);
    hipFree(d_arrayB);

    return 0;
}