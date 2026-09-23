#include <alpaka/alpaka.hpp>
#include <alpaka/graph/cuda/GraphCudaRt.hpp>

#include <cstddef>
#include <iostream>

int main()
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;

    auto graph = alpaka::createGraph<Acc>();
    std::cout << "Graph created\n";

    alpaka::destroyGraph(graph);
    std::cout << "Graph destroyed\n";
}