#pragma once

namespace alpaka
{
    namespace trait
    {
        //! Maps accelerator type to its graph type.
        template <typename TAcc, typename TSfinae = void>
        struct GraphType;

        //! Creates graph for an accelerator.
        template <typename TAcc, typename TSfinae = void>
        struct CreateGraph;

        //! Destroys graph.
        template <typename TGraph, typename TSfinae = void>
        struct DestroyGraph;
    } // namespace trait

    //! Public graph type alias.
    template <typename TAcc>
    using Graph = typename trait::GraphType<TAcc>::type;

    //! Public graph creation function.
    template <typename TAcc>
    auto createGraph() -> Graph<TAcc>
    {
        return trait::CreateGraph<TAcc>::createGraph();
    }

    //! Public graph destruction function.
    template <typename TGraph>
    auto destroyGraph(TGraph const &graph) -> void
    {
        trait::DestroyGraph<TGraph>::destroyGraph(graph);
    }

} // namespace alpaka