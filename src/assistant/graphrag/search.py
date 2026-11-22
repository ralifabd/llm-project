"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from collections import deque
import networkx as nx
import re
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase, MAP_SCHEMA
LOCAL_FACTS_LIMIT = 8000

def _answer_model_failure_ranking(question: str, kb: KnowledgeBase) -> Answer | None:
    """Отвечает на вопрос о модели с наибольшим числом отказов."""
    if not re.search(r"больше всего отказ|наибольш\w*\s+количеств\w*\s+отказ", question, re.IGNORECASE):
        return None

    line_names = sorted({
        name for entity_type, name in find_names(question, kb.vocabulary) if entity_type == "ProductLine"
    })

    if len(line_names) != 1:
        return None

    graph = kb.graph
    line_id = f"ProductLine:{line_names[0]}"

    if line_id not in graph:
        return None

    models = []

    for model_id, _, edge_data in graph.in_edges(line_id, data=True):
        if (graph.nodes[model_id].get("type") == "Model" and edge_data.get("type") == "BELONGS_TO"):
            models.append(model_id)

    if not models:
        return None

    counts = {}

    for model_id in models:
        reports = set()

        for _, component_id, edge_data in graph.out_edges(model_id, data=True):
            if (edge_data.get("type") != "FAILED" or graph.nodes[component_id].get("type") != "Component"):
                continue
            for chunk_id in edge_data.get("sources", []):
                document_id = chunk_id.split("#", 1)[0]

                if document_id.startswith("report_"):
                    reports.add(document_id)
        counts[model_id] = reports

    max_count = max(len(reports) for reports in counts.values())

    if max_count == 0:
        return None

    winners = sorted(model_id for model_id, reports in counts.items() if len(reports) == max_count)

    if len(winners) == 1:
        answer_text = (
            f"Больше всего отказов в линейке "
            f"{line_names[0]} у модели {winners[0]}: "
            f"{max_count} уникальных отчётов об отказах."
        )
    else:
        answer_text = (
            f"В линейке {line_names[0]} одинаковое максимальное "
            f"количество отказов у моделей "
            f"{', '.join(winners)}: {max_count} отчётов."
        )

    sources = sorted({report for model_id in winners for report in counts[model_id]})
    facts = [f"{model_id}: {len(reports)} уникальных отчётов" for model_id, reports in sorted(counts.items())]

    return Answer(text=answer_text, sources=sources, facts=facts, refused=False)


def _community_sources(community, graph: nx.DiGraph) -> list[str]:
    """Возвращает источники узлов и граничных связей сообщества."""
    node_ids = set(community.nodes)
    sources = set()

    for node_id in node_ids:
        if node_id not in graph:
            continue

        sources.update(graph.nodes[node_id].get("sources", []))

        for _, _, data in graph.in_edges(node_id, data=True):
            sources.update(data.get("sources", []))

        for _, _, data in graph.out_edges(node_id, data=True):
            sources.update(data.get("sources", []))

    return sorted(sources)


def _answer_model_product_line(question: str, kb: KnowledgeBase) -> Answer | None:
    """Отвечает на вопрос о принадлежности модели к линейке."""
    if not re.search(r"в какую линейку входит|к какой линейке относится", question, re.IGNORECASE):
        return None

    anchors = _collect_local_anchors(question, kb)
    models = [node for node in anchors if node.startswith("Model:")]

    if len(models) != 1:
        return None

    model_id = models[0]

    if model_id not in kb.graph:
        return None

    graph = kb.graph

    product_lines = [
        (target, data) for _, target, data in graph.out_edges(model_id, data=True)
        if data.get("type") == "BELONGS_TO" and graph.nodes[target].get("type") == "ProductLine"
    ]

    if len(product_lines) != 1:
        return None

    line_id, edge_data = product_lines[0]
    line_name = line_id.split(":", 1)[1]

    sources = sorted({source.split("#", 1)[0] for source in edge_data.get("sources", [])})

    return Answer(
        text=f"Модель {model_id.split(':', 1)[1]} входит в линейку «{line_name}».",
        sources=sources,
        facts=[f"{model_id} --BELONGS_TO--> {line_id}"],
        refused=False,
    )



def _select_global_communities(question: str, kb: KnowledgeBase) -> list:
    anchors = _collect_local_anchors(question, kb)

    if anchors is None:
        return []

    specific_anchors = [anchor for anchor in anchors if not anchor.startswith("ProductLine:")]

    if not specific_anchors:
        return [c for c in kb.communities if c.summary]

    graph = kb.graph
    undirected = graph.to_undirected()
    anchor_distances = {}

    for anchor in specific_anchors:
        if anchor not in undirected:
            continue

        anchor_distances[anchor] = nx.single_source_shortest_path_length(undirected, anchor, cutoff=3)

    ranked = []

    for community in kb.communities:
        if not community.summary:
            continue

        community_nodes = set(community.nodes)
        score = 0
        matched_anchors = 0

        for distances in anchor_distances.values():
            reachable = [distances[node] for node in community_nodes if node in distances]

            if not reachable:
                continue

            distance = min(reachable)
            matched_anchors += 1
            score += 4 - distance

        if matched_anchors:
            ranked.append((matched_anchors, score, community))

    ranked.sort(key=lambda item: (-item[0], -item[1], str(item[2].community_id)))
    return [item[2] for item in ranked[:5]]


def _collect_local_anchors(question: str, kb: KnowledgeBase) -> list[str] | None:
    """Находит опорные сущности без обращения к языковой модели."""
    graph = kb.graph
    anchors = []

    for code in find_codes(question):
        node_id = f"{code_type(code)}:{code}"

        if node_id not in graph:
            return None

        if node_id not in anchors:
            anchors.append(node_id)

    for entity_type, name in find_names(question, kb.vocabulary):
        node_id = f"{entity_type}:{name}"

        if node_id in graph and node_id not in anchors:
            anchors.append(node_id)

    return anchors



def _collect_local_edges(graph: nx.DiGraph, anchors: list[str]) -> set[tuple[str, str]]:
    """Собирает связи между якорями или их окрестность."""
    undirected = graph.to_undirected()
    selected_edges = set()

    if len(anchors) >= 2:
        for index, source in enumerate(anchors):
            for target in anchors[index + 1:]:
                for path in nx.all_simple_paths(undirected, source, target, cutoff=3):
                    for left, right in zip(path, path[1:]):
                        if graph.has_edge(left, right):
                            selected_edges.add((left, right))
                        if graph.has_edge(right, left):
                            selected_edges.add((right, left))

    elif len(anchors) == 1:
        start = anchors[0]
        distances = {start: 0}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            if distances[current] >= 2:
                continue

            for neighbour in undirected.neighbors(current):
                if neighbour not in distances:
                    distances[neighbour] = distances[current] + 1
                    queue.append(neighbour)

        # Собираем входящие и исходящие рёбра окрестности.
        nearby = set(distances)

        for node in nearby:
            for source, target in graph.in_edges(node):
                if source in nearby:
                    selected_edges.add((source, target))

            for source, target in graph.out_edges(node):
                if target in nearby:
                    selected_edges.add((source, target))

    return selected_edges

def _extend_replacement_chains(graph: nx.DiGraph, selected_edges: set[tuple[str, str]]) -> None:
    """Добавляет цепочки замен и поставщиков конечных деталей."""
    parts = set()
    for source, target in selected_edges:
        if graph.nodes[source].get("type") == "Part":
            parts.add(source)
        if graph.nodes[target].get("type") == "Part":
            parts.add(target)

    visited = set()
    queue = deque(parts)
    latest_parts = set()

    while queue:
        part = queue.popleft()
        if part in visited:
            continue

        visited.add(part)

        replacements = [
            target for _, target, data in graph.out_edges(part, data=True) if data.get("type") == "REPLACED_BY"
        ]

        if replacements:
            for target in replacements:
                selected_edges.add((part, target))
                queue.append(target)
        else:
            latest_parts.add(part)

    # Добавляем поставщиков только конечных деталей цепочки.
    for part in latest_parts:
        for _, target, data in graph.out_edges(part, data=True):
            if data.get("type") == "SUPPLIED_BY":
                selected_edges.add((part, target))



def _format_local_facts(graph: nx.DiGraph, selected_edges: set[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Формирует факты и источники в пределах лимита 8000 символов."""
    facts = []
    sources = []
    total_length = 0

    for source, target, data in graph.edges(data=True):
        if (source, target) not in selected_edges:
            continue

        relation = data.get("type", "UNKNOWN")
        weight = data.get("weight", 1)

        fact = f"{source} --{relation}--> {target} (вес: {weight})"
        prefix_length = len(str(len(facts) + 1)) + 2
        extra_length = len(fact) + prefix_length

        if facts:
            extra_length += 1

        if total_length + extra_length > LOCAL_FACTS_LIMIT:
            continue

        facts.append(fact)
        total_length += extra_length

        for chunk_id in data.get("sources", []):
            document_id = chunk_id.split("#", 1)[0]

            if document_id:
                sources.append(document_id)

    return facts, sorted(set(sources))

def _answer_discontinued_parts(question: str, kb: KnowledgeBase) -> Answer | None:
    """Находит детали линейки, для которых указана замена."""
    if not re.search(r"снят\w*\s+с\s+производств|снят\w*\s+с\s+поставк",question, re.IGNORECASE):
        return None

    line_names = {
        name for entity_type, name in find_names(question, kb.vocabulary) if entity_type == "ProductLine"
    }

    if len(line_names) != 1:
        return None

    line_id = f"ProductLine:{next(iter(line_names))}"

    if line_id not in kb.graph:
        return None

    graph = kb.graph
    models = [
        source for source, _, data in graph.in_edges(line_id, data=True)
        if data.get("type") == "BELONGS_TO" and graph.nodes[source].get("type") == "Model"
    ]

    discontinued = {}

    for model_id in models:
        for _, part_id, data in graph.out_edges(model_id, data=True):
            if (data.get("type") != "HAS_PART" or graph.nodes[part_id].get("type") != "Part"):
                continue

            for _, replacement_id, replacement_data in graph.out_edges(part_id, data=True):
                if replacement_data.get("type") != "REPLACED_BY":
                    continue

                discontinued[part_id] = {
                    "replacement": replacement_id,
                    "sources": set(data.get("sources", [])) | set(replacement_data.get("sources", [])),
                }

    if not discontinued:
        return None

    lines = ["В линейке найдены детали, для которых указана замена:"]

    sources = set()

    for part_id, info in sorted(discontinued.items()):
        lines.append(f"{part_id} заменена на {info['replacement']}.")
        sources.update(source.split("#", 1)[0] for source in info["sources"])

    return Answer(
        text="\n".join(lines),
        sources=sorted(sources),
        facts=[
            f"{part_id} --REPLACED_BY--> {info['replacement']}"
            for part_id, info in sorted(discontinued.items())
        ],
        refused=False,
    )

def local_search( question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

       Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
       При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
       и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
       до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
       Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
       `Answer(..., refused=True)` без вызова модели.
       """
    graph = kb.graph
    anchors = _collect_local_anchors(question, kb)

    if not anchors:
        return Answer(text="", refused=True)

    selected_edges = _collect_local_edges(graph, anchors)
    _extend_replacement_chains(graph, selected_edges)
    facts, sources = _format_local_facts(graph, selected_edges)

    if not facts:
        return Answer(text="", refused=True)

    numbered_facts = "\n".join(f"{index}. {fact}" for index, fact in enumerate(facts, start=1))

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — сервисный ассистент по холодильному оборудованию. "
                "Отвечай на русском языке. Используй исключительно факты "
                "из предоставленного списка. Не добавляй сведения от себя "
                "и не делай предположений. Если фактов недостаточно, "
                "честно сообщи об этом. "
                "При наличии цепочки REPLACED_BY укажи последнюю деталь "
                "в цепочке. Не путай вес связи с количеством отказов."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Вопрос: {question}\n\n"
                f"Опорные сущности: {', '.join(anchors)}\n\n"
                f"Факты из графа:\n{numbered_facts}\n\n"
                "Ответь только на основании этих фактов."
            ),
        },
    ]

    response = llm.chat(messages, max_tokens=500)
    answer_text = response.content.strip()

    if not answer_text:
        return Answer(text="", refused=True)

    return Answer(
        text=answer_text,
        sources=sources,
        facts=facts,
        refused=False,
    )


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

        Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
        Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
        Если таких ответов нет, возвращается `Answer(..., refused=True)`.
        """
    line_answer = _answer_model_product_line(question, kb)

    if line_answer is not None:
        return line_answer

    discontinued_answer = _answer_discontinued_parts(question, kb)

    if discontinued_answer is not None:
        return discontinued_answer

    stats_answer = _answer_model_failure_ranking(question, kb)

    if stats_answer is not None:
        return stats_answer

    anchors = _collect_local_anchors(question, kb)
    specific_anchors = [anchor for anchor in (anchors or []) if not anchor.startswith("ProductLine:")]

    if specific_anchors:
        local_answer = local_search(question, kb, llm)
        if not local_answer.refused:
            return local_answer

    partial_answers = []
    selected_communities = _select_global_communities(question, kb)

    for community in selected_communities:
        if not community.summary:
            continue

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты анализируешь сводку одного сообщества графа знаний "
                    "о холодильном оборудовании. Определи, содержит ли "
                    "сводка сведения для ответа на вопрос. "
                    "Используй только сведения из данной сводки. "
                    "Отвечай только тогда, когда ответ прямо подтверждается "
                    "сводкой. Не путай вес связи с количеством отказов. "
                    "Не переноси сведения между разными моделями, "
                    "компонентами и линейками. Не придумывай числа и связи. "
                    "Если данных недостаточно, верни пустой answer и score 0. "
                    "Если ответ подтверждён, дай краткий конкретный ответ "
                    "и оцени релевантность от 1 до 10. "
                    "Верни JSON с полями answer и score."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\n"
                    f"Сводка сообщества:\n{community.summary}"
                ),
            },
        ]

        try:
            response = llm.chat(messages, json_schema=MAP_SCHEMA)
            result = json.loads(response.content)

            if not isinstance(result, dict):
                continue

            answer_text = result.get("answer", "")
            score = result.get("score", 0)

            if (not isinstance(answer_text, str) or type(score) is not int or not 1 <= score <= 10):
                continue

            answer_text = answer_text.strip()

            if not answer_text:
                continue

            sources = {
                chunk_id.split("#", 1)[0] for chunk_id in _community_sources(community, kb.graph)
            }

            partial_answers.append({
                "community_id": community.community_id,
                "score": score,
                "answer": answer_text,
                "sources": sorted(sources),
            })

        except (json.JSONDecodeError, AttributeError, TypeError):
            continue

    if not partial_answers:
        return Answer(text="", refused=True)

    partial_answers.sort(key=lambda item: item["score"], reverse=True)

    partial_text = "\n\n".join(
        f"Сообщество {item['community_id']} "
        f"(релевантность {item['score']}/10):\n"
        f"{item['answer']}" for item in partial_answers
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — сервисный ассистент по холодильному оборудованию. "
                "Тебе переданы частичные ответы из разных сообществ графа. "
                "Составь на их основе один ответ. "
                "Используй только факты из частичных ответов. "
                "Объединяй дополняющие друг друга сведения. "
                "Не добавляй новых сущностей, кодов, чисел и связей. "
                "Не переноси свойства одной модели или линейки на другую. "
                "Если ответы противоречат друг другу, укажи на противоречие. "
                "Сохраняй точные обозначения моделей и деталей. "
                "Отвечай кратко, по существу и на русском языке. "
                "Верни только текст ответа."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Вопрос: {question}\n\n"
                f"Частичные ответы:\n{partial_text}\n\n"
                "Сформулируй итоговый ответ."
            ),
        },
    ]

    response = llm.chat(messages, max_tokens=500)
    final_text = response.content.strip()

    try:
        result = json.loads(final_text)
        if isinstance(result, dict):
            final_text = result.get("answer", "")
            if not isinstance(final_text, str):
                final_text = ""
            final_text = final_text.strip()
    except json.JSONDecodeError:
        pass

    if not final_text:
        return Answer(text="", refused=True)

    sources = sorted({source for item in partial_answers for source in item["sources"]})

    return Answer(
        text=final_text,
        sources=sources,
        facts=[item["answer"] for item in partial_answers],
        refused=False
    )
