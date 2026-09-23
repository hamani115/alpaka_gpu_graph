#pragma once

#include <alpaka/graph/Traits.hpp>

#include <cuda_runtime_api.h>

#include <stdexcept>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

namespace alpaka
{
    namespace detail
    {
        class GraphCudaRt
        {
        public:
            explicit GraphCudaRt(cudaGraph_t graph)
                : m_graph{graph}
            {
            }

            auto getNativeHandle() const -> cudaGraph_t
            {
                return m_graph;
            }

        private:
            cudaGraph_t m_graph;
        };

    } // namespace detail

    namespace trait
    {
        template <typename TDim, typename TIdx>
        struct GraphType<AccGpuCudaRt<TDim, TIdx>>
        {
            using type = ::alpaka::detail::GraphCudaRt;
        };

        template <typename TDim, typename TIdx>
        struct CreateGraph<AccGpuCudaRt<TDim, TIdx>>
        {
            static auto createGraph() -> ::alpaka::detail::GraphCudaRt
            {
                cudaGraph_t nativeGraph{};

                auto const result = cudaGraphCreate(&nativeGraph, 0u);

                if (result != cudaSuccess)
                {
                    throw std::runtime_error{cudaGetErrorString(result)};
                }

                return ::alpaka::detail::GraphCudaRt{nativeGraph};
            }
        };

        template <>
        struct DestroyGraph<alpaka::detail::GraphCudaRt>
        {
            static auto destroyGraph(alpaka::detail::GraphCudaRt const &graph) -> void
            {
                auto const result = cudaGraphDestroy(graph.getNativeHandle());

                if (result != cudaSuccess)
                {
                    throw std::runtime_error{cudaGetErrorString(result)};
                }
            }
        };

    } // namespace trait
} // namespace alpaka

#endif