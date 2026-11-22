"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART_ID_PATTERN = re.compile(r"P-\d{4}[A-Z]?")

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Поиск конкретных сведений в графе знаний.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Вопрос о конкретной детали, модели или ошибке."}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Обзорный поиск и обобщение сведений по теме.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Обзорный вопрос по базе знаний."}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Проверить остаток детали на указанном складе.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали, например P-1042B."},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                },
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Получить цену и сведения о детали из ERP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали, например P-1042B."}
                },
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создать заказ на деталь для указанной модели и склада.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string"},
                    "warehouse": {"type": "string", "enum": WAREHOUSES},
                    "qty": {"type": "integer", "minimum": 1},
                    "model": {"type": "string"},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    def valid_part_id(part_id: str) -> bool:
        return isinstance(part_id, str) and PART_ID_PATTERN.fullmatch(part_id) is not None

    def search_knowledge(question: str = "") -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Необходимо указать вопрос."}

        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": answer.facts,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def summarize_topic(question: str = "") -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Необходимо указать вопрос."}

        answer = global_search(question, kb, llm)
        return {
            "answer": answer.text,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def check_stock(part_id: str = "", warehouse: str = "") -> dict:
        if not valid_part_id(part_id):
            return {"error": "Некорректный номер детали. Ожидается формат P-1042 или P-1042B."}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад: {warehouse}."}

        try:
            return erp.get_stock(part_id, warehouse)
        except ErpNotFound as exc:
            return {"error": f"Деталь не найдена: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    def get_price(part_id: str = "") -> dict:
        if not valid_part_id(part_id):
            return {"error": "Некорректный номер детали. Ожидается формат P-1042 или P-1042B."}

        try:
            return erp.get_price(part_id)
        except ErpNotFound as exc:
            return {"error": f"Деталь не найдена: {exc}", "error_type": "not_found"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    def create_service_order(part_id: str = "", warehouse: str = "", qty: int = 0, model: str = "") -> dict:
        if not valid_part_id(part_id):
            return {"error": "Некорректный номер детали. Ожидается формат P-1042 или P-1042B."}
        if warehouse not in WAREHOUSES:
            return {"error": f"Неизвестный склад: {warehouse}."}
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            return {"error": "Количество должно быть целым числом больше нуля."}
        if not isinstance(model, str) or not model.strip():
            return {"error": "Необходимо указать модель оборудования."}

        try:
            return erp.create_order(part_id, warehouse, qty, model)
        except ErpNotFound as exc:
            return {"error": f"Данные для заказа не найдены: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила заказ: {exc}"}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
