#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#if defined(GPU_GRAPH_ENABLE_CUDA) && defined(GPU_GRAPH_ENABLE_HIP)
#error Enable either CUDA or HIP, not both in the same translation unit.
#endif

#if defined(GPU_GRAPH_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(GPU_GRAPH_ENABLE_HIP)
#include <hip/hip_runtime.h>
#else
#error Define GPU_GRAPH_ENABLE_CUDA or GPU_GRAPH_ENABLE_HIP.
#endif

namespace gpu_graph
{
    // Backend tags. These contain no data.
    struct CudaBackend
    {
    };

    struct HipBackend
    {
    };

    // Backend-independent replacement for CUDA/HIP dim3
    struct Dim3
    {
        unsigned int x{1u};
        unsigned int y{1u};
        unsigned int z{1u};
    };

    // Information needed to create one kernel node.
    struct KernelNodeConfig
    {
        // ! I believe this is not where to define them, check fill implementation
        void *function{};
        Dim3 grid{};
        Dim3 block{};
        void **arguments{};
        std::size_t sharedMemoryBytes{0u};

        KernelNodeConfig(
            void *kernelFunction,
            Dim3 gridDimensions,
            Dim3 blockDimensions,
            void **kernelArguments,
            std::size_t dynamicSharedMemoryBytes = 0u)
            : function{kernelFunction}, grid{gridDimensions}, block{blockDimensions}, arguments{kernelArguments}, sharedMemoryBytes{dynamicSharedMemoryBytes}
        {
        }
    };

    /*
     * Primary template.
     *
     * It intentionally has no implementation.
     * Only supported backends receive specializations.
     */
    template <typename TBackend>
    struct Api;

#if defined(GPU_GRAPH_ENABLE_CUDA)

    template <>
    struct Api<CudaBackend>
    {
        using Error = cudaError_t;
        using Graph = cudaGraph_t;
        using Node = cudaGraphNode_t;
        using Executable = cudaGraphExec_t;
        using Stream = cudaStream_t;

        //! Use cudaCheck without the second parameter better
        static auto check(
            Error const error,
            char const *const operation) -> void
        {
            if (error != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string{operation} + " failed: " + cudaGetErrorString(error));
            }
        }

        //! will be replaced by queue
        static auto createStream() -> Stream
        {
            Stream stream{};

            check(
                cudaStreamCreate(&stream),
                "cudaStreamCreate");

            return stream;
        }

        static auto createGraph() -> Graph
        {
            Graph graph{};

            check(
                cudaGraphCreate(&graph, 0u),
                "cudaGraphCreate");

            return graph;
        }

        //! This has to be changed so it is simplier to how CUDA adds kernel to a graph
        static auto addKernelNode(
            Graph const graph,
            std::span<Node const> const dependencies,
            KernelNodeConfig const &config) -> Node
        {
            cudaKernelNodeParams params{};

            params.func = config.function;

            params.gridDim = dim3{
                config.grid.x,
                config.grid.y,
                config.grid.z};

            params.blockDim = dim3{
                config.block.x,
                config.block.y,
                config.block.z};

            params.sharedMemBytes = config.sharedMemoryBytes;
            params.kernelParams = config.arguments;
            params.extra = nullptr;

            Node node{};

            Node const *const dependencyPtr = dependencies.empty()
                                                  ? nullptr
                                                  : dependencies.data();

            check(
                cudaGraphAddKernelNode(
                    &node,
                    graph,
                    dependencyPtr,
                    dependencies.size(),
                    &params),
                "cudaGraphAddKernelNode");

            return node;
        }

        //! probably change name to make it like alpaka stile, so graphInstantiate
        static auto instantiate(Graph const graph) -> Executable
        {
            Executable executable{};

            check(
                cudaGraphInstantiate(
                    &executable,
                    graph,
                    nullptr,
                    nullptr,
                    0u),
                "cudaGraphInstantiate");

            return executable;
        }

        static auto launch(
            //! again, you can not just call it executable, it has to be sepcific to graphs
            Executable const executable,
            Stream const stream) -> void
        {
            check(
                cudaGraphLaunch(executable, stream),
                "cudaGraphLaunch");
        }

        static auto synchronize(Stream const stream) -> void
        {
            check(
                cudaStreamSynchronize(stream),
                "cudaStreamSynchronize");
        }

        static auto destroyExecutable(
            Executable const executable) -> void
        {
            check(
                cudaGraphExecDestroy(executable),
                "cudaGraphExecDestroy");
        }

        static auto destroyGraph(Graph const graph) -> void
        {
            check(
                cudaGraphDestroy(graph),
                "cudaGraphDestroy");
        }

        static auto destroyStream(Stream const stream) -> void
        {
            check(
                cudaStreamDestroy(stream),
                "cudaStreamDestroy");
        }
    };

#endif

#if defined(GPU_GRAPH_ENABLE_HIP)

    template <>
    struct Api<HipBackend>
    {
        using Error = hipError_t;
        using Graph = hipGraph_t;
        using Node = hipGraphNode_t;
        using Executable = hipGraphExec_t;
        using Stream = hipStream_t;

        static auto check(
            Error const error,
            char const *const operation) -> void
        {
            if (error != hipSuccess)
            {
                throw std::runtime_error(
                    std::string{operation} + " failed: " + hipGetErrorString(error));
            }
        }

        static auto createStream() -> Stream
        {
            Stream stream{};

            check(
                hipStreamCreate(&stream),
                "hipStreamCreate");

            return stream;
        }

        static auto createGraph() -> Graph
        {
            Graph graph{};

            check(
                hipGraphCreate(&graph, 0u),
                "hipGraphCreate");

            return graph;
        }

        static auto addKernelNode(
            Graph const graph,
            std::span<Node const> const dependencies,
            KernelNodeConfig const &config) -> Node
        {
            hipKernelNodeParams params{};

            params.func = config.function;

            params.gridDim = dim3{
                config.grid.x,
                config.grid.y,
                config.grid.z};

            params.blockDim = dim3{
                config.block.x,
                config.block.y,
                config.block.z};

            params.sharedMemBytes = config.sharedMemoryBytes;
            params.kernelParams = config.arguments;
            params.extra = nullptr;

            Node node{};

            Node const *const dependencyPtr = dependencies.empty()
                                                  ? nullptr
                                                  : dependencies.data();

            check(
                hipGraphAddKernelNode(
                    &node,
                    graph,
                    dependencyPtr,
                    dependencies.size(),
                    &params),
                "hipGraphAddKernelNode");

            return node;
        }

        static auto instantiate(Graph const graph) -> Executable
        {
            Executable executable{};

            check(
                hipGraphInstantiate(
                    &executable,
                    graph,
                    nullptr,
                    nullptr,
                    0u),
                "hipGraphInstantiate");

            return executable;
        }

        static auto launch(
            Executable const executable,
            Stream const stream) -> void
        {
            check(
                hipGraphLaunch(executable, stream),
                "hipGraphLaunch");
        }

        static auto synchronize(Stream const stream) -> void
        {
            check(
                hipStreamSynchronize(stream),
                "hipStreamSynchronize");
        }

        static auto destroyExecutable(
            Executable const executable) -> void
        {
            check(
                hipGraphExecDestroy(executable),
                "hipGraphExecDestroy");
        }

        static auto destroyGraph(Graph const graph) -> void
        {
            check(
                hipGraphDestroy(graph),
                "hipGraphDestroy");
        }

        static auto destroyStream(Stream const stream) -> void
        {
            check(
                hipStreamDestroy(stream),
                "hipStreamDestroy");
        }
    };

#endif

    // Public type aliases

    template <typename TBackend>
    using Graph = typename Api<TBackend>::Graph;

    template <typename TBackend>
    using Node = typename Api<TBackend>::Node;

    template <typename TBackend>
    using Executable = typename Api<TBackend>::Executable;

    template <typename TBackend>
    using Stream = typename Api<TBackend>::Stream;

    // Public functions

    template <typename TBackend>
    auto createStream() -> Stream<TBackend>
    {
        return Api<TBackend>::createStream();
    }

    template <typename TBackend>
    auto createGraph() -> Graph<TBackend>
    {
        return Api<TBackend>::createGraph();
    }

    template <typename TBackend>
    auto addKernelNode(
        Graph<TBackend> const graph,
        std::span<Node<TBackend> const> const dependencies,
        KernelNodeConfig const &config) -> Node<TBackend>
    {
        return Api<TBackend>::addKernelNode(
            graph,
            dependencies,
            config);
    }

    // Convenience overload for a node with no dependencies.
    template <typename TBackend>
    auto addKernelNode(
        Graph<TBackend> const graph,
        KernelNodeConfig const &config) -> Node<TBackend>
    {
        return addKernelNode<TBackend>(
            graph,
            std::span<Node<TBackend> const>{},
            config);
    }

    template <typename TBackend>
    auto instantiate(
        Graph<TBackend> const graph) -> Executable<TBackend>
    {
        return Api<TBackend>::instantiate(graph);
    }

    template <typename TBackend>
    auto launch(
        Executable<TBackend> const executable,
        Stream<TBackend> const stream) -> void
    {
        Api<TBackend>::launch(executable, stream);
    }

    //! already exists in alpaka for alpaka::queue
    template <typename TBackend>
    auto synchronize(Stream<TBackend> const stream) -> void
    {
        Api<TBackend>::synchronize(stream);
    }

    template <typename TBackend>
    auto destroyExecutable(
        Executable<TBackend> const executable) -> void
    {
        Api<TBackend>::destroyExecutable(executable);
    }

    //! should we do it automatically like for alpaka::queue?
    //! probably not because this has to be explicit in case more nodes added on existing graph
    template <typename TBackend>
    auto destroyGraph(Graph<TBackend> const graph) -> void
    {
        Api<TBackend>::destroyGraph(graph);
    }

    //? alpaka automatically do it?
    template <typename TBackend>
    auto destroyStream(Stream<TBackend> const stream) -> void
    {
        Api<TBackend>::destroyStream(stream);
    }
}