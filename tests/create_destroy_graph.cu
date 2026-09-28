#include <alpaka/alpaka.hpp>
#include <alpaka/graph/cuda/GraphCudaRt.hpp>

#include <cstddef>
#include <iostream>
#include <typeinfo>

int main()
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;

    auto graph = alpaka::createGraph<Acc>();
    std::out << "Graph created!" << '\n';

    std::cout << "Graph type: " << typeid(graph).name() << '\n';

    alpaka::destroyGraph(graph);
    std::cout << "Graph destroyed" << '\n';
}