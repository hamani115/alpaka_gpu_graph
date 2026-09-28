#pragma once

namespace alpaka
{
    namespace trait
    {
        template <typename TAcc, typename TSfinae = void>
        struct GraphType;

        //! Creates graph for an accelerator
        template <typename TAcc, typename TSfinae = void>
        struct CreateGraph;

        template <typename TGraph, typename TSfinae = void>
        struct DestroyGraph;

        template <typename TAcc, typename TSfinae = void>
        struct GraphNodeType;
    } // namespace trait

    //! Public graph type alias
    template <typename TAcc>
    using Graph = typename trait::GraphType<TAcc>::type;

    template <typename TAcc>
    using GraphNode = typename trait::GraphNodeType<TAcc>::type;

    //! Public functions
    template <typename TAcc>
    auto createGraph() -> Graph<TAcc>
    {
        return trait::CreateGraph<TAcc>::createGraph();
    }

    template <typename TGraph>
    auto destroyGraph(TGraph const &graph) -> void
    {
        trait::DestroyGraph<TGraph>::destroyGraph(graph);
    }

} // namespace alpaka