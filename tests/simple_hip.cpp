#include <gpu_graph/Graph.hpp>

#include <array>

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
    // Change only to HipBackend for the HIP build.

    auto stream = gpu_graph::createStream<Backend>();
    auto graph = gpu_graph::createGraph<Backend>();

    unsigned int const threadsPerBlock = 256u;

    unsigned int const blocksPerGrid = static_cast<unsigned int>(
        (n + threadsPerBlock - 1u) / threadsPerBlock);

    float *d_input{};
    float *d_output{};
    std::size_t n{};

    // Add node

    void *addArguments[] = {
        &d_input,
        &d_output,
        &n};

    gpu_graph::KernelNodeConfig const addConfig{
        reinterpret_cast<void *>(addKernel),
        {blocksPerGrid, 1u, 1u},
        {threadsPerBlock, 1u, 1u},
        addArguments};

    auto const addNode =
        gpu_graph::addKernelNode<Backend>(
            graph,
            addConfig);

    // Multiply node

    void *multiplyArguments[] = {
        &d_output,
        &n};

    gpu_graph::KernelNodeConfig const multiplyConfig{
        reinterpret_cast<void *>(multiplyKernel),
        {blocksPerGrid, 1u, 1u},
        {threadsPerBlock, 1u, 1u},
        multiplyArguments};

    std::array<gpu_graph::Node<Backend>, 1u> const multiplyDependencies{
        addNode};

    auto const multiplyNode =
        gpu_graph::addKernelNode<Backend>(
            graph,
            multiplyDependencies,
            multiplyConfig);

    // Assignment node

    void *assignArguments[] = {
        &d_output,
        &n};

    gpu_graph::KernelNodeConfig const assignConfig{
        reinterpret_cast<void *>(assignKernel),
        {blocksPerGrid, 1u, 1u},
        {threadsPerBlock, 1u, 1u},
        assignArguments};

    std::array<gpu_graph::Node<Backend>, 1u> const assignDependencies{
        multiplyNode};

    auto const assignNode =
        gpu_graph::addKernelNode<Backend>(
            graph,
            assignDependencies,
            assignConfig);

    auto executable =
        gpu_graph::instantiate<Backend>(graph);

    for (std::size_t iteration = 0u;
         iteration < numberOfIterations;
         ++iteration)
    {
        gpu_graph::launch<Backend>(
            executable,
            stream);
    }

    gpu_graph::synchronize<Backend>(stream);

    gpu_graph::destroyExecutable<Backend>(executable);
    gpu_graph::destroyGraph<Backend>(graph);
    gpu_graph::destroyStream<Backend>(stream);
}