#include <gpu_graph/Graph.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <cuda_runtime.h>

__global__ void addKernel(double* arrayA, const int* arrayB, std::size_t size)
{
    const std::size_t x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < size)
    {
        arrayA[x] += arrayB[x];
    }
}

__global__ void multiplyKernel(double* arrayA, std::size_t size)
{
    const std::size_t x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < size)
    {
        arrayA[x] *= 2.0;
    }
}

__global__ void assignKernel(int* arrayB, std::size_t size)
{
    const std::size_t x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < size)
    {
        arrayB[x] = 3;
    }
}

int main()
{
    using Backend = gpu_graph::CudaBackend;

    auto stream = gpu_graph::createStream<Backend>();
    auto graph = gpu_graph::createGraph<Backend>();

    unsigned int const numberOfIterations = 10;

    std::size_t n{4};

    // Managed memory so that both CPU and GPU can access it.
    double* d_arrayA{};
    int* d_arrayB{};

    cudaMallocManaged(&d_arrayA, n * sizeof(double));
    cudaMallocManaged(&d_arrayB, n * sizeof(int));

    // Initialize the arrays on the CPU.
    for (std::size_t i = 0; i < n; ++i)
    {
        d_arrayA[i] = static_cast<double>(i);
        d_arrayB[i] = 0;
    }

    unsigned int const threadsPerBlock = 256u;

    unsigned int const blocksPerGrid =
        static_cast<unsigned int>(
            (n + threadsPerBlock - 1u) / threadsPerBlock);

    // ============================================================
    // Node 1: assignKernel
    //
    // arrayB[x] = 3
    // ============================================================

    void* assignArguments[] = {
        &d_arrayB,
        &n
    };

    gpu_graph::KernelNodeConfig const assignConfig{
        reinterpret_cast<void*>(assignKernel),
        {blocksPerGrid, 1u, 1u},
        {threadsPerBlock, 1u, 1u},
        assignArguments
    };

    auto const assignNode =
        gpu_graph::addKernelNode<Backend>(
            graph,
            assignConfig);

    // ============================================================
    // Node 2: addKernel
    //
    // arrayA[x] += arrayB[x]
    //
    // Depends on assignKernel.
    // ============================================================

    void* addArguments[] = {
        &d_arrayA,
        &d_arrayB,
        &n
    };

    gpu_graph::KernelNodeConfig const addConfig{
        reinterpret_cast<void*>(addKernel),
        {blocksPerGrid, 1u, 1u},
        {threadsPerBlock, 1u, 1u},
        addArguments
    };

    std::array<gpu_graph::Node<Backend>, 1u> const addDependencies{
        assignNode
    };

    auto const addNode =
        gpu_graph::addKernelNode<Backend>(
            graph,
            addDependencies,
            addConfig);

    // ============================================================
    // Node 3: multiplyKernel
    //
    // arrayA[x] *= 2
    //
    // Depends on addKernel.
    // ============================================================

    void* multiplyArguments[] = {
        &d_arrayA,
        &n
    };

    gpu_graph::KernelNodeConfig const multiplyConfig{
        reinterpret_cast<void*>(multiplyKernel),
        {blocksPerGrid, 1u, 1u},
        {threadsPerBlock, 1u, 1u},
        multiplyArguments
    };

    std::array<gpu_graph::Node<Backend>, 1u> const multiplyDependencies{
        addNode
    };

    auto const multiplyNode =
        gpu_graph::addKernelNode<Backend>(
            graph,
            multiplyDependencies,
            multiplyConfig);

    // Avoid unused-variable warning.
    (void)multiplyNode;

    // ============================================================
    // Instantiate graph
    // ============================================================

    auto executable =
        gpu_graph::instantiate<Backend>(graph);

    // ============================================================
    // Launch graph multiple times
    // ============================================================

    for (std::size_t iteration = 0;
         iteration < numberOfIterations;
         ++iteration)
    {
        gpu_graph::launch<Backend>(
            executable,
            stream);
    }

    gpu_graph::synchronize<Backend>(stream);

    // ============================================================
    // Print results
    // ============================================================

    std::cout << "arrayA:" << std::endl;

    for (std::size_t i = 0; i < n; ++i)
    {
        std::cout
            << "arrayA[" << i << "] = "
            << d_arrayA[i]
            << std::endl;
    }

    std::cout << "\narrayB:" << std::endl;

    for (std::size_t i = 0; i < n; ++i)
    {
        std::cout
            << "arrayB[" << i << "] = "
            << d_arrayB[i]
            << std::endl;
    }

    // ============================================================
    // Cleanup
    // ============================================================

    gpu_graph::destroyExecutable<Backend>(executable);
    gpu_graph::destroyGraph<Backend>(graph);
    gpu_graph::destroyStream<Backend>(stream);

    cudaFree(d_arrayA);
    cudaFree(d_arrayB);

    return 0;
}
