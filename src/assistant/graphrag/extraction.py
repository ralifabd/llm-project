"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Chunk, ExtractionResult, ENTITY_TYPES, EXTRACTION_SCHEMA, RELATION_SIGNATURES, Relation, Entity
from assistant.infra.config import vocabulary
from assistant.infra.grounding import (canonical_name, find_codes, find_names)
import json

SYSTEM = {
    "Ты ассистент сервисной службы компании «ПолярТех». Отвечай на вопрос только по фрагментам документов ниже. "
    "Твоя задача извлекать сущности и связи из технического текста. Верни только корректный JSON по заданной схеме. "
    "Не придумывай факты. "
    "Направление связей:\n"
    "BELONGS_TO: Model -> ProductLine\n"
    "HAS_PART: Model -> Part\n"
    "IS_A: Part -> Component\n"
    "INDICATES: ErrorCode -> Component\n"
    "FIXED_BY: ErrorCode -> Procedure\n"
    "REPLACED_BY: Part -> Part\n"
    "SUPPLIED_BY: Part -> Supplier\n"
    "FAILED: Model -> Component\n"
    "Пример: если P-1042 заменена на P-1042B то верни P-1042 REPLACED_BY P-1042B."
}


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю
    result = ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
    words = vocabulary()
    codes = set(find_codes(chunk.text))
    names = set(find_names(chunk.text, words))

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": chunk.text}]

    data = None
    for _ in range(2):
        try:
            response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
            data = json.loads(response.content)

            if not isinstance(data, dict):
                raise ValueError("Не корректный JSON-объект")

            if not isinstance(data.get("entities"), list) or not isinstance(data.get("relations"), list):
                raise ValueError("Некорректные entities или relations")

            for item in data["entities"]:
                if (not isinstance(item, dict) or not isinstance(item.get("type"), str)
                        or not isinstance(item.get("name"), str)):
                    raise ValueError("Некорректная сущность")

            fields = ("subject_type", "subject", "type", "object_type", "object")
            for item in data["relations"]:
                if not isinstance(item, dict) or any(not isinstance(item.get(field), str) for field in fields):
                    raise ValueError("Некорректная связь")
            break
        except Exception:
            data = None

    if data is None:
        return result

    entities = {}

    def get_entity(entity_type: str, raw_name: str) -> Entity | None:
        if entity_type not in ENTITY_TYPES:
            return None

        name = canonical_name(entity_type, raw_name, words)
        if name is None:
            return None

        if entity_type in {"Model", "Part", "ErrorCode", "Procedure"}:
            if name not in codes:
                return None
        elif (entity_type, name) not in names:
            return None

        entity = Entity(entity_type, name)
        entities[entity.node_id] = entity
        return entity

    for item in data["entities"]:
        if get_entity(item["type"], item["name"]) is None:
            result.dropped += 1

    relations = {}
    for item in data["relations"]:
        relation_type = item["type"]

        if (relation_type not in RELATION_SIGNATURES or RELATION_SIGNATURES[relation_type]
                != (item["subject_type"], item["object_type"])):
            result.dropped += 1
            continue

        subject = get_entity(item["subject_type"], item["subject"])
        obj = get_entity(item["object_type"], item["object"])

        if subject is None or obj is None:
            result.dropped += 1
            continue

        relation = Relation(subject, relation_type, obj)
        if not relation.matches_schema():
            result.dropped += 1
            continue

        key = (subject.node_id, relation_type, obj.node_id)
        relations[key] = relation

    result.entities = list(entities.values())
    result.relations = list(relations.values())
    return result
