"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
import re

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

SYSTEM = (
    "Ты извлекаешь сущности и связи из технических документов сервисной службы компании «ПолярТех».\n"
    "Верни только JSON по заданной схеме. Не добавляй факты, которых нет в тексте.\n\n"

    "Допустимые типы связей. Порядок сущностей СТРОГО важен:\n"
    "BELONGS_TO: Model -> ProductLine\n"
    "HAS_PART: Model -> Part\n"
    "FAILED: Model -> Component\n"
    "IS_A: Part -> Component\n"
    "REPLACED_BY: Part -> Part\n"
    "SUPPLIED_BY: Part -> Supplier\n"
    "INDICATES: ErrorCode -> Component\n"
    "FIXED_BY: ErrorCode -> Procedure\n\n"

    "ОСОБОЕ ПРАВИЛО ДЛЯ FIXED_BY:\n"
    "1. Если в документе сказано, что регламент или процедура применяется при определённой ошибке, извлекай FIXED_BY.\n"
    "2. Если сказано «Регламент SP-104 применяется при ошибке E-17», создай связь E-17 FIXED_BY SP-104.\n"
    "3. Если процедура описывает устранение конкретной ошибки, связывай ErrorCode с этой "
    "Procedure отношением FIXED_BY.\n"
    "4. Направление строгое: сначала ErrorCode, затем Procedure.\n"
    "5. Не создавай FIXED_BY, если связь ошибки с процедурой не подтверждается текстом.\n\n"

    "ОСОБОЕ ПРАВИЛО ДЛЯ FAILED:\n"
    "1. Если в отчёте указана модель оборудования и описан отказ конкретного узла, обязательно извлекай связь FAILED.\n"
    "2. Направление строгое: Model -> Component.\n"
    "3. Например, текст «Оборудование BZ-120. Диагностика показала "
    "отказ узла: вентилятор испарителя» означает:\n"
    "   BZ-120 FAILED вентилятор испарителя.\n"
    "4. Если также указан код ошибки E-18, извлекай отдельную связь:\n"
    "   E-18 INDICATES вентилятор испарителя.\n"
    "5. Если указаны обе связи, извлекай ОБЕ. INDICATES не заменяет FAILED.\n"
    "6. Не создавай FAILED только на основании упоминания компонента "
    "или кода ошибки. Нужен подтверждённый текстом отказ узла.\n"
    "7. Не используй HAS_PART или IS_A вместо FAILED.\n\n"

    "Правила:\n"
    "1. Для каждой связи сначала указывай исходную сущность, "
    "затем целевую — строго в порядке выше.\n"
    "2. Не меняй местами source и target.\n"
    "3. Код E-xx — это ErrorCode, а не Component или Model.\n"
    "4. Код P-xxxx — это Part, а не Component или Supplier.\n"
    "5. Название узла, например «компрессор», — это Component.\n"
    "6. Название поставщика — это Supplier.\n"
    "7. Не создавай связь, если её тип или направление "
    "не соответствуют допустимому списку.\n"
    "8. Учитывай контекст заголовков и вводных предложений. Если написано «В AR-210 установлены следующие узлы», "
    "то каждая деталь P-xxxx в последующем списке связана с моделью AR-210 отношением HAS_PART.\n"
    "9. Для каждой такой детали извлекай обе связи отдельно: "
    "Model HAS_PART Part и Part IS_A Component. Не заменяй HAS_PART связью IS_A."
    "ВАЖНО: если перед списком деталей указано, что они установлены "
    "в модели (например, «В AR-210 установлены следующие узлы»), "
    "создай HAS_PART от этой модели к КАЖДОЙ детали из списка. "
    "Это отдельная связь, не заменяющая IS_A и SUPPLIED_BY. "
    "Для списка из 8 деталей должны быть созданы 8 отношений HAS_PART.\n\n"

    "Примеры:\n"
    "Если модель AR-210 относится к линейке «Арктика»:\n"
    "AR-210 BELONGS_TO Арктика.\n"
    "Если деталь P-1010 является компрессором:\n"
    "P-1010 IS_A компрессор.\n"
    "Если деталь P-1010 поставляет «КомпрессорМаш»:\n"
    "P-1010 SUPPLIED_BY КомпрессорМаш.\n"
    "Если код E-11 указывает на неисправность компрессора:\n"
    "E-11 INDICATES компрессор.\n"
    "Если в тексте сказано «Деталь P-1042 снята с производства. "
    "Вместо неё устанавливается деталь P-1042B», извлекай:\n"
    "P-1042 REPLACED_BY P-1042B.\n"
    "Первой всегда указывай снятую с производства (старую) деталь, "
    "второй — устанавливаемую вместо неё (новую). "
    "Не меняй направление этой связи на обратное.\n"
    "Если в тексте сказано «В AR-210 установлены следующие узлы», "
    "а в списке указана деталь P-1010:\n"
    "AR-210 HAS_PART P-1010.\n"
    "Если в тексте сказано «Регламент SP-104 применяется "
    "при ошибке E-17»:\n"
    "E-17 FIXED_BY SP-104.\n"
)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    result = ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
    words = vocabulary()
    codes = set(find_codes(chunk.text))
    names = set(find_names(chunk.text, words))

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": chunk.text}]

    data = None
    for _ in range(2):
        try:
            response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=8192)
            data = json.loads(response.content)

            if not isinstance(data, dict):
                raise ValueError("Некорректный JSON-объект")

            if not isinstance(data.get("entities"), list) or not isinstance(data.get("relations"), list):
                raise ValueError("Некорректные entities или relations")

            for item in data["entities"]:
                if (not isinstance(item, dict) or not isinstance(item.get("type"), str)
                        or not isinstance(item.get("name"), str)):
                    raise ValueError("Некорректная сущность")

            fields = ("subject_type", "subject", "type", "object_type", "object")
            for item in data["relations"]:
                if (not isinstance(item, dict) or any(not isinstance(item.get(field), str) for field in fields)):
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
    explicit_replacements = _explicit_replacement_pairs(chunk.text)
    for item in data["relations"]:
        relation_type = item["type"]
        expected = RELATION_SIGNATURES.get(relation_type)

        if expected is None:
            result.dropped += 1
            continue

        actual = (item["subject_type"], item["object_type"])

        if actual == expected:
            pass
        elif actual == expected[::-1] and expected[0] != expected[1]:
            item = {
                **item,
                "subject_type": item["object_type"],
                "subject": item["object"],
                "object_type": item["subject_type"],
                "object": item["subject"],
            }
        else:
            result.dropped += 1
            continue

        if relation_type == "REPLACED_BY":
            pair = (item["subject"].upper(), item["object"].upper())
            if (pair[1], pair[0]) in explicit_replacements:
                item = {**item, "subject": item["object"], "object": item["subject"]}

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

    pattern = re.compile(r"деталь\s+(P-\d{4}[A-Z]?)\s*\(([^()\n]+)\)", re.IGNORECASE)

    llm_is_a_parts = {item["subject"].upper() for item in data["relations"]
        if item.get("type") == "IS_A" and item.get("subject_type") == "Part"
    }

    for part_code, component_name in pattern.findall(chunk.text):
        if part_code.upper() in llm_is_a_parts:
            continue
        part = get_entity("Part", part_code)
        component = get_entity("Component", component_name.strip())

        if part is None or component is None:
            continue

        relation = Relation(part, "IS_A", component)
        if relation.matches_schema():
            key = (part.node_id, "IS_A", component.node_id)
            relations[key] = relation

    result.entities = list(entities.values())
    result.relations = list(relations.values())
    return result


def _explicit_replacement_pairs(text: str) -> set[tuple[str, str]]:
    """Находит явно указанные пары старая деталь -> новая деталь."""
    pattern = re.compile(
        r"деталь\s+(P-\d{4}[A-Z]?)\b"
        r"[^.\n]*снята\s+с\s+производства\s*[.!?]?\s*"
        r"вместо\s+(?:неё|нее)\s+устанавливается\s+деталь\s+"
        r"(P-\d{4}[A-Z]?)\b", re.IGNORECASE,
    )
    return {(old.upper(), new.upper()) for old, new in pattern.findall(text)}
