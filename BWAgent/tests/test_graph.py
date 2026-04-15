import pytest

from graph.workflow import build_graph, compiled_graph


def test_build_graph_contains_nodes() -> None:
    graph = build_graph()
    assert graph is not None
    assert hasattr(graph, "nodes") or hasattr(graph, "get_node")


def test_compiled_graph_has_stream_interface() -> None:
    assert compiled_graph is not None
    assert hasattr(compiled_graph, "stream")
    assert hasattr(compiled_graph, "ainvoke") or hasattr(compiled_graph, "invoke")


@pytest.mark.parametrize("node_name", ["router_node", "planner_node", "researcher_node", "writer_node"])
def test_graph_has_expected_nodes(node_name: str) -> None:
    graph = build_graph()
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        node_names = list(nodes.keys())
    elif isinstance(nodes, list):
        node_names = [getattr(node, "name", None) for node in nodes if node is not None]
    else:
        node_names = []
    assert node_name in node_names
