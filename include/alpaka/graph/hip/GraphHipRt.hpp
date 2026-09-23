#pragma once

#include <alpaka/graph/Traits.hpp>

#include <hip/hip_runtime_api.h>

#include <stdexcept>

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka
{
    namespace detail
    {
        class GraphHipRt
        {
        public:
            explicit GraphHipRt(hipGraph_t graph)
                : m_graph{graph}
            {
            }

            auto getNativeHandle() const -> hipGraph_t
            {
                return m_graph;
            }

        private:
            hipGraph_t m_graph;
        };

    } // namespace detail

    namespace trait
    {
        template<typename TDim, typename TIdx>
        struct GraphType<AccGpuHipRt<TDim,TIdx>>
        {
            using type = ::alpaka::detail::GraphHipRt;
        };

        template<typename TDim, typename TIdx>
        struct CreateGraph<AccGpuHipRt<TDim,TIdx>>
        {
            static auto createGraph() -> ::alpaka::detail::GraphHipRt
            {
                hipGraph_t nativeGraph{};

                auto const result = hipGraphCreate(&nativeGraph, 0u);

                if(result != hipSuccess)
                {
                    throw std::runtime_error(hipGetErrorString(result))
                }

                reutrn ::alpaka::detail::GraphHipRt{nativeGraph};
            }
        }

        template <>
        struct DestoryGraph<::alpaka::detail::GraphHipRt>
        {
            static auto destoryGraph(::alpaka::detail::GraphHipRt const& graph) -> void
            {
                auto const result = hipGraphDestroy(graph.getNativeHandle());

                if (result != hipSuccess)
                {
                    throw std::runtime_error(hipGetErrorString(result));
                }
            }
        }
        
    } // namespace trait
} // namespace alpaka

#endif