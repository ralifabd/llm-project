
"""Сборка графа знаний. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф."""
    graph = nx.DiGraph()
    node_sources = {}
    edge_sources = {}

    def add_entity(entity, chunk_id):
        node_id = entity.node_id

        if node_id not in graph:
            graph.add_node(node_id, type=entity.type, name=entity.name, sources=[])
            node_sources[node_id] = set()
        elif (
            graph.nodes[node_id].get("type") != entity.type or graph.nodes[node_id].get("name") != entity.name):
            raise ValueError(f"Конфликт атрибутов узла: {node_id}")

        node_sources[node_id].add(chunk_id)

    for result in results:
        chunk_id = result.chunk_id

        for entity in result.entities:
            add_entity(entity, chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            subject = relation.subject
            obj = relation.object
            relation_type = relation.type
            add_entity(subject, chunk_id)
            add_entity(obj, chunk_id)
            edge_key = (subject.node_id, obj.node_id, relation_type)
            edge_sources.setdefault(edge_key, set()).add(chunk_id)

    for node_id, sources in node_sources.items():
        graph.nodes[node_id]["sources"] = sorted(sources)

    for (source, target, relation_type), sources in edge_sources.items():
        existing = graph.get_edge_data(source, target)

        if existing is not None and existing.get("type") != relation_type:
            raise ValueError(
                f"Несколько типов связей для пары {source} -> {target}: {existing.get('type')} и {relation_type}"
            )

        graph.add_edge(source, target, type=relation_type, weight=len(sources), sources=sorted(sources))

    conflicts = 0
    checked_pairs = set()

    for source, target, data in list(graph.edges(data=True)):
        if data.get("type") != "REPLACED_BY":
            continue

        reverse = graph.get_edge_data(target, source)

        if reverse is None or reverse.get("type") != "REPLACED_BY":
            continue

        pair = frozenset((source, target))
        if pair in checked_pairs:
            continue

        checked_pairs.add(pair)
        conflicts += 1

        forward_weight = data["weight"]
        reverse_weight = reverse["weight"]

        if forward_weight > reverse_weight:
            graph.remove_edge(target, source)
        elif reverse_weight > forward_weight:
            graph.remove_edge(source, target)
        else:
            graph.remove_edge(source, target)
            graph.remove_edge(target, source)

    graph.graph["conflicts"] = conflicts
    return graph
