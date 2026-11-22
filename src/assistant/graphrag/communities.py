"""Сообщества графа и их сводки. Требование FR-3."""

import json
from collections import defaultdict

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SUMMARY_WORD_LIMIT = 150

def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if graph.number_of_nodes() == 0:
        return []

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes(data=True))

    for source, target, data in graph.edges(data=True):
        weight = float(data.get("weight", 1))

        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    groups = nx.community.louvain_communities(undirected, weight="weight", seed=42)
    normalized = [sorted(group) for group in groups]
    normalized.sort(key=lambda nodes: nodes[0])
    return [Community(community_id=index, nodes=nodes) for index, nodes in enumerate(normalized)]

def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    node_ids = set(community.nodes)
    failure_stats = _failure_stats_by_product_line(graph)
    community_failure_stats = {line_id: counts for line_id, counts in failure_stats.items() if line_id in node_ids}

    nodes = []

    for node_id in sorted(node_ids):
        if node_id not in graph:
            continue

        data = graph.nodes[node_id]
        nodes.append({
            "id": node_id,
            "type": data.get("type", ""),
            "name": data.get("name", node_id),
        })

    node_counts_by_type = defaultdict(int)

    for node in nodes:
        node_counts_by_type[node["type"]] += 1

    internal_edges = []
    boundary_edges = []
    external_nodes = {}
    internal_weights = defaultdict(int)
    boundary_weights = defaultdict(int)

    for source, target, data in graph.edges(data=True):
        source_inside = source in node_ids
        target_inside = target in node_ids

        if not source_inside and not target_inside:
            continue

        relation_type = data.get("type", "UNKNOWN")
        weight = data.get("weight", 1)

        edge = {
            "source": source,
            "type": relation_type,
            "target": target,
            "weight": weight,
        }

        if source_inside and target_inside:
            internal_edges.append(edge)
            internal_weights[relation_type] += weight
            continue

        boundary_edges.append(edge)
        boundary_weights[relation_type] += weight

        external_id = target if source_inside else source

        if external_id in graph:
            external_data = graph.nodes[external_id]
            external_nodes[external_id] = {
                "id": external_id,
                "type": external_data.get("type", ""),
                "name": external_data.get("name", external_id),
            }

    facts = {
        "community_id": community.community_id,
        "nodes": nodes,
        "internal_edges": internal_edges,
        "boundary_edges": boundary_edges,
        "external_nodes": list(external_nodes.values()),
        "statistics": {
            "node_count": len(nodes),
            "internal_edge_count": len(internal_edges),
            "boundary_edge_count": len(boundary_edges),
            "node_count_by_type": dict(sorted(node_counts_by_type.items())),
            "failure_counts_by_component": community_failure_stats,
            "internal_weight_by_relation": dict(sorted(internal_weights.items())),
            "boundary_weight_by_relation": dict(sorted(boundary_weights.items())),
        },
    }

    messages = [
        {
            "role": "system",
            "content": (
                "Ты составляешь краткие сводки сообществ графа знаний "
                "о холодильном оборудовании. Используй только переданные факты. "
                "Опиши основные сущности и связи сообщества. "
                "Перечисляй внутренние сущности только из списка nodes. "
                "Внешние сущности упоминай отдельно и только для объяснения "
                "граничных связей. Отличай внутренние связи от граничных. "
                "Количество сущностей каждого типа бери только из "
                "statistics.node_count_by_type. "
                "Не используй веса связей как количество сущностей или отказов. "
                "Не рассчитывай статистику отказов самостоятельно. "
                "Не придумывай сущности, связи и числа. "
                "Пиши по-русски. "
                "Сводка должна быть краткой."
            ),
        },
        {
            "role": "user",
            "content": (
                "Составь сводку сообщества по следующим фактам:\n"
                + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
            ),
        },
    ]

    response = llm.chat(messages)
    summary = response.content.strip()
    stats_text = _build_failure_stats_text(community_failure_stats, graph)
    return _limit_summary_words(summary, stats_text)

def _build_failure_stats_text(community_failure_stats: dict, graph: nx.DiGraph,) -> str:
    """Формирует текст статистики отказов из рассчитанных кодом значений."""
    lines = []

    for line_id, counts in sorted(community_failure_stats.items()):
        if not counts:
            continue

        line_name = graph.nodes[line_id].get("name", line_id)
        details = "; ".join(f"{component}: {count} отчётов"
            for component, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )

        lines.append(f"Статистика отказов линейки {line_name}: {details}.")
    return "\n".join(lines)


def _limit_summary_words(summary: str, stats_text: str) -> str:
    """Ограничивает сводку 150 словами, отдавая приоритет статистике."""
    stats_words = stats_text.split()
    summary_words = summary.split()

    if len(stats_words) >= SUMMARY_WORD_LIMIT:
        return " ".join(stats_words[:SUMMARY_WORD_LIMIT])

    available_words = SUMMARY_WORD_LIMIT - len(stats_words)
    summary_part = " ".join(summary_words[:available_words])

    if summary_part and stats_text:
        return f"{summary_part}\n{stats_text}"

    return summary_part or stats_text

def _failure_stats_by_model(graph: nx.DiGraph) -> dict[str, int]:
    """Считает уникальные отчёты об отказах для каждой модели."""
    result = {}

    for model_id, data in graph.nodes(data=True):
        if data.get("type") != "Model":
            continue
        reports = set()

        for _, component_id, edge_data in graph.out_edges(model_id, data=True):
            if (edge_data.get("type") != "FAILED" or graph.nodes[component_id].get("type") != "Component"):
                continue
            for chunk_id in edge_data.get("sources", []):
                document_id = chunk_id.split("#", 1)[0]
                if document_id.startswith("report_"):
                    reports.add(document_id)
        result[model_id] = len(reports)

    return result

def _failure_stats_by_product_line(graph: nx.DiGraph) -> dict:
    """Считает уникальные отчёты об отказах компонентов для каждой линейки."""
    models_by_line = defaultdict(set)

    for source, target, data in graph.edges(data=True):
        if (data.get("type") == "BELONGS_TO" and graph.nodes[source].get("type") == "Model"
                and graph.nodes[target].get("type") == "ProductLine"):
            models_by_line[target].add(source)

    result = {}

    for line_id, models in models_by_line.items():
        reports_by_component = defaultdict(set)

        for model_id in models:
            for _, component_id, data in graph.out_edges(model_id, data=True):
                if (data.get("type") != "FAILED" or graph.nodes[component_id].get("type") != "Component"):
                    continue

                for chunk_id in data.get("sources", []):
                    document_id = chunk_id.split("#", 1)[0]
                    if document_id.startswith("report_"):
                        reports_by_component[component_id].add(document_id)

        result[line_id] = {
            graph.nodes[component_id].get("name", component_id): len(reports)
            for component_id, reports in sorted(reports_by_component.items())
        }

    return result