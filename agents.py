"""LLM-backed service investigators and coordinator."""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from tools import CHECKOUT_TOOLS, PAYMENT_TOOLS, SHIPPING_TOOLS

load_dotenv()

MODEL_CONFIG = {
    "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com",
    "temperature": 0,
}

MAX_TOOL_CALLS = 5


class SREState(TypedDict):
    incident: str
    checkout_report: str | None
    payment_report: str | None
    shipping_report: str | None
    final_diagnosis: str | None


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        **MODEL_CONFIG,
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )


def _investigate(service: str, incident: str, service_tools: list[BaseTool]) -> str:
    model = _model()
    model_with_tools = model.bind_tools(service_tools)
    tools_by_name = {service_tool.name: service_tool for service_tool in service_tools}
    messages = [
        SystemMessage(
            content=f"""You are the SRE agent responsible only for {service}.
Investigate independently using only the supplied {service} tools.
Call one tool at a time and make at most {MAX_TOOL_CALLS} tool calls.
Do not invent information.

When you have enough evidence, return a concise report with:
- service health status
- observed anomaly
- relevant evidence
- suspected dependency, if any
- confidence"""
        ),
        HumanMessage(content=f"Incident: {incident}\nInvestigate it yourself."),
    ]
    tool_calls_used = 0

    while tool_calls_used < MAX_TOOL_CALLS:
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            if tool_calls_used >= MAX_TOOL_CALLS:
                result = "Tool-call limit reached. This call was not executed."
            else:
                selected_tool = tools_by_name[tool_call["name"]]
                result = selected_tool.invoke(tool_call["args"])
                tool_calls_used += 1
                print(
                    f"\n[{service}] Action: {tool_call['name']}({tool_call['args']})"
                    f"\n[{service}] Observation:\n{result}"
                )

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    messages.append(
        HumanMessage(content="The tool-call limit is reached. Return your report now.")
    )
    return model.invoke(messages).content


def checkout_agent(state: SREState) -> dict[str, str]:
    report = _investigate(
        "Checkout",
        state["incident"],
        CHECKOUT_TOOLS,
    )
    return {"checkout_report": report}


def payment_agent(state: SREState) -> dict[str, str]:
    report = _investigate(
        "Payment",
        state["incident"],
        PAYMENT_TOOLS,
    )
    return {"payment_report": report}


def shipping_agent(state: SREState) -> dict[str, str]:
    report = _investigate(
        "Shipping",
        state["incident"],
        SHIPPING_TOOLS,
    )
    return {"shipping_report": report}


def coordinator(state: SREState) -> dict[str, str]:
    prompt = f"""You are the incident coordinator. Synthesize the three service reports.

Incident: {state["incident"]}

Checkout report:
{state["checkout_report"]}

Payment report:
{state["payment_report"]}

Shipping report:
{state["shipping_report"]}

Return a concise final diagnosis with:
- root cause
- responsible component or boundary
- supporting evidence
- confidence

Use only the reports above. Do not invent information."""
    return {"final_diagnosis": _model().invoke(prompt).content}
