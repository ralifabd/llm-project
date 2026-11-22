"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

PART_PATTERN = re.compile(r"\bP-\d{4}[A-Z]?\b", re.IGNORECASE)
PRICE_KEYWORDS = re.compile(r"(стоит|цен[аы]|сколько.*руб|price)", re.IGNORECASE)
ORDER_KEYWORDS = re.compile(r"(оформ|заказ|закажи|купи|покуп)", re.IGNORECASE)

def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Ты сервисный ассистент по холодильному оборудованию.\n"
                "\n"
                "ОБЩИЕ ПРАВИЛА:\n"
                "- Используй инструменты для получения фактов.\n"
                "- Не выдумывай номера деталей, цены и остатки.\n"
                "- Не считай код ошибки номером детали.\n"
                "- Не считай название модели или компонента номером детали.\n"
                "\n"
                "ПОИСК ДЕТАЛЕЙ:\n"
                "- Если пользователь спрашивает о детали по модели, компоненту "
                "или коду ошибки, сначала вызови search_knowledge.\n"
                "- Дождись результата поиска, прежде чем вызывать ERP-инструменты.\n"
                "- Используй только номера деталей P-XXXX или P-XXXXA, "
                "подтверждённые результатами поиска или вопросом пользователя.\n"
                "- Если найдена цепочка REPLACED_BY, используй последнюю "
                "деталь в цепочке.\n"
                "- Не выбирай деталь только по совпадению названия или модели.\n\n"
                
                "ПОИСК РЕГЛАМЕНТОВ И ПОСТАВЩИКОВ:\n"
                "- Если пользователь спрашивает, по какому регламенту устраняется\n"
                "  ошибка или отказ компонента, используй search_knowledge.\n"
                "- Не используй summarize_topic для поиска конкретного регламента,\n"
                "  процедуры или инструкции.\n"
                "- Если пользователь указывает конкретный номер детали P-XXXX\n"
                "  и спрашивает её поставщика, сначала вызови search_knowledge.\n"
                "- Если пользователь спрашивает о поставщике, цене или остатках\n"
                "  детали, сначала установи её номер и актуальность.\n"
                "- Для прямого номера детали не придумывай сведения:\n"
                "  используй только подтверждённые связи графа.\n"

                "РАБОТА С ERP:\n"
                "- Для цены используй get_price только после определения детали.\n"
                "- Для остатков используй check_stock только после определения детали.\n"
                "- Для заказа сначала определи актуальную деталь через search_knowledge.\n"
                "- В create_service_order передавай номер детали, а не код ошибки.\n"
                "- Модель, склад и количество должны соответствовать запросу пользователя.\n"
                "- Не создавай заказ без подтверждения пользователя: вызывай create_service_order, "
                "агент сам запросит подтверждение и вернёт номер детали в ответе.\n\n"
                
                "ОБЗОРНЫЕ ВОПРОСЫ:\n"
                "- Для обзора темы используй summarize_topic.\n"
                "- Не разрешай противоречия догадками или по названию модели.\n\n"
                "Если инструмент не нашёл подтверждённых сведений, сообщи об этом."
            ),
        },
        {"role": "user", "content": question},
    ]

    steps = []
    sources = []
    cache = {}
    known_parts = set(PART_PATTERN.findall(question))
    refused = False
    pending_action = None
    tool_calls_count = 0
    get_price_called = False
    order_confirmation_needed = False

    qty_match = re.search(r"\b(\d+)\s*(?:шт\.?|штук(?:и|у|а)?|единиц(?:ы|у)?)\b", question, re.IGNORECASE)
    user_qty = int(qty_match.group(1)) if qty_match else None
    asks_price = bool(PRICE_KEYWORDS.search(question))
    _ = bool(ORDER_KEYWORDS.search(question))

    while True:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())

        if not response.tool_calls:
            answer = response.content.strip()
            break

        if tool_calls_count >= max_steps:
            answer = (
                response.content.strip()
                or "Достигнут лимит вызовов инструментов. Не удалось завершить проверку."
            )
            break

        for call in response.tool_calls:
            args = call.arguments
            name = call.name
            error = None
            result = None

            if tool_calls_count >= max_steps:
                break

            tool_calls_count += 1

            if name not in tools:
                error = f"Неизвестный инструмент: {name}"

            elif not isinstance(args, dict) or "_raw" in args:
                error = "Некорректные аргументы вызова инструмента."

            elif ("part_id" in args and args["part_id"] not in known_parts):
                error = (
                    f"Номер детали {args['part_id']} не встречался в вопросе "
                    f"или результатах предыдущих инструментов."
                )

            elif name == "create_service_order":
                if user_qty is None or args.get("qty") != user_qty:
                    error = "Количество не указано пользователем. Уточни количество."
                elif not confirm:
                    pending_action = {"tool": name, "args": args}
                    order_confirmation_needed = True
                    error = "Для создания заказа требуется подтверждение пользователя."
                    step = ToolStep(tool=name, args=args, error=error)
                    steps.append(step)
                    messages.append(tool_message(call, {"error": error}))
                    continue

            if name == "get_price":
                get_price_called = True

            if error is None:
                key = (name, tuple(sorted(args.items())))

                if key in cache:
                    result = cache[key]
                else:
                    attempts = 3 if name in {"check_stock", "get_price", "create_service_order"} else 1

                    for attempt in range(attempts):
                        try:
                            result = tools[name](**args)
                            break
                        except ErpUnavailable as exc:
                            if attempt == attempts - 1:
                                error = str(exc)
                        except Exception as exc:
                            error = str(exc)
                            break

                    if error is None:
                        cache[key] = result

            if error is not None:
                step = ToolStep(tool=name, args=args, error=error)
                tool_result = {"error": error}
            else:
                step = ToolStep(tool=name, args=args, result=result)
                tool_result = result

                if isinstance(result, dict):
                    known_parts.update(PART_PATTERN.findall(str(result)))
                    sources.extend(result.get("sources", []))
                    if result.get("refused") or result.get("error_type") == "not_found":
                        refused = True

            steps.append(step)
            messages.append(tool_message(call, tool_result))

        if tool_calls_count >= max_steps:
            continue

        if asks_price and not get_price_called and known_parts:
            part_id = next(iter(known_parts))
            messages.append({
                "role": "user",
                "content": f"Деталь {part_id} найдена. Теперь вызови get_price для получения цены."
            })
            continue

        if order_confirmation_needed and pending_action:
            messages.append({
                "role": "user",
                "content": (
                    "Заказ подготовлен, но требуется подтверждение пользователя. "
                    "Объясни пользователю, что заказ готов к оформлению (номер детали, склад, количество, модель), "
                    "и что для завершения нужно подтверждение. Не отказывай в ответе."
                )
            })
            continue

    def is_grounded(text: str) -> bool:
        answer_parts = set(PART_PATTERN.findall(text))
        if not answer_parts.issubset(known_parts):
            return False

        prices = {str(step.result["price_rub"]) for step in steps if step.result and "price_rub" in step.result}
        stocks = {str(step.result["qty"]) for step in steps if step.result and "qty" in step.result}
        answer_prices = re.findall(r"(\d[\d\s]*)\s*(?:руб\.?|₽)", text, re.IGNORECASE)

        for price in answer_prices:
            if re.sub(r"\s", "", price) not in prices:
                return False

        answer_stocks = re.findall(r"\b(\d+)\s*(?:шт\.?|штук(?:и|у|а)?)", text, re.IGNORECASE)
        if any(qty not in stocks for qty in answer_stocks):
            return False

        return True

    if answer and not is_grounded(answer):
        messages.append({
            "role": "user",
            "content": (
                "Проверь свой предыдущий ответ. В нём есть данные, "
                "которые не подтверждаются вопросом или результатами "
                "инструментов. Исправь ответ, используя только "
                "подтверждённые сведения."
            ),
        })

        correction = llm.chat(messages, tools=TOOL_SCHEMAS)
        answer = correction.content.strip()
        if not answer or not is_grounded(answer):
            if order_confirmation_needed and pending_action:
                answer = (
                    f"Заказ готов к оформлению: деталь {pending_action['args']['part_id']}, "
                    f"склад {pending_action['args']['warehouse']}, "
                    f"количество {pending_action['args']['qty']}, "
                    f"модель {pending_action['args']['model']}. "
                    f"Для завершения оформления требуется ваше подтверждение."
                )
            else:
                answer = "Не могу дать достоверный ответ: данные не подтверждены."
                refused = True

    return AgentResult(
        answer=answer,
        steps=steps,
        sources=list(dict.fromkeys(sources)),
        refused=refused,
        needs_confirmation=pending_action is not None,
        pending_action=pending_action,
    )
