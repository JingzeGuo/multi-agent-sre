"""LLM-backed service investigators and coordinator."""

import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from tools import CHECKOUT_TOOLS, PAYMENT_TOOLS, SHIPPING_TOOLS

load_dotenv()

MODEL_CONFIG = {
    "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com",
    "temperature": 0,
}

INITIAL_TOOL_CALLS = 5
FOLLOWUP_TOOL_CALLS = 2

AgentName = Literal["checkout", "payment", "shipping"]


class FollowUpRequest(TypedDict):
    agent: AgentName
    question: str


class CoordinatorAssessment(TypedDict):
    status: Literal["sufficient", "needs_followup"]
    followups: list[FollowUpRequest]


class FollowUpState(TypedDict):
    incident: str
    agent: AgentName
    previous_report: str
    question: str


class SREState(TypedDict):
    incident: str
    checkout_report: str | None
    payment_report: str | None
    shipping_report: str | None
    followups: list[FollowUpRequest]
    followup_reports: Annotated[dict[str, str], operator.or_]
    final_diagnosis: str | None


def _model(*, thinking: bool = True) -> ChatOpenAI:
    return ChatOpenAI(
        **MODEL_CONFIG,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        extra_body={
            "thinking": {"type": "enabled" if thinking else "disabled"}
        },
    )


def _run_tool_loop(
    service: str,
    messages: list[BaseMessage],
    service_tools: list[BaseTool],
    max_tool_calls: int,
) -> str:
    model = _model()
    model_with_tools = model.bind_tools(service_tools)
    tools_by_name = {service_tool.name: service_tool for service_tool in service_tools}
    tool_calls_used = 0

    while tool_calls_used < max_tool_calls:
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            if tool_calls_used >= max_tool_calls:
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


def _investigate(service: str, incident: str, service_tools: list[BaseTool]) -> str:
    messages = [
        SystemMessage(
            content=f"""You are the SRE agent responsible only for {service}.
Investigate independently using only the supplied {service} tools.
Call one tool at a time and make at most {INITIAL_TOOL_CALLS} tool calls.
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
    return _run_tool_loop(service, messages, service_tools, INITIAL_TOOL_CALLS)


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


def follow_up(state: FollowUpState) -> dict[str, dict[str, str]]:
    service, service_tools = {
        "checkout": ("Checkout", CHECKOUT_TOOLS),
        "payment": ("Payment", PAYMENT_TOOLS),
        "shipping": ("Shipping", SHIPPING_TOOLS),
    }[state["agent"]]
    messages = [
        SystemMessage(
            content=f"""You are the {service} SRE agent in targeted follow-up mode.
Answer only the coordinator's question using your previous report and, if needed,
at most {FOLLOWUP_TOOL_CALLS} additional calls to your own {service} tools.
Do not invent information or investigate other services."""
        ),
        HumanMessage(
            content=f"""Original incident: {state['incident']}

Your previous report:
{state['previous_report']}

Coordinator question:
{state['question']}"""
        ),
    ]
    report = _run_tool_loop(
        service,
        messages,
        service_tools,
        FOLLOWUP_TOOL_CALLS,
    )
    return {"followup_reports": {state["agent"]: report}}


def coordinator_assess(state: SREState) -> dict[str, list[FollowUpRequest]]:
    prompt = f"""You are the incident coordinator. Assess the initial reports.

Incident: {state["incident"]}

Checkout report:
{state["checkout_report"]}

Payment report:
{state["payment_report"]}

Shipping report:
{state["shipping_report"]}

Decide whether the evidence is sufficient for a well-supported root-cause diagnosis.
If it is insufficient, request targeted local evidence from at most two distinct
agents. Ask at most one specific, tool-answerable question per selected agent.
Agents can inspect only their current status, recent logs, and recent traces;
do not ask for metrics, configuration, or unavailable historical data.
Do not request follow-up merely to repeat evidence already present."""
    assessment = (
        _model(thinking=False)
        .with_structured_output(
            CoordinatorAssessment,
            method="function_calling",
        )
        .invoke(prompt)
    )

    followups: list[FollowUpRequest] = []
    selected_agents: set[str] = set()
    if assessment["status"] == "needs_followup":
        for request in assessment["followups"]:
            if request["agent"] in selected_agents or not request["question"].strip():
                continue
            followups.append(request)
            selected_agents.add(request["agent"])
            if len(followups) == 2:
                break

    return {"followups": followups}


def final_coordinator(state: SREState) -> dict[str, str]:
    followup_reports = state.get("followup_reports", {})
    followup_text = "\n\n".join(
        f"{agent.title()} follow-up report:\n{report}"
        for agent, report in followup_reports.items()
    ) or "No follow-up was requested."
    prompt = f"""You are the incident coordinator. Produce the final diagnosis.

Incident: {state["incident"]}

Initial Checkout report:
{state["checkout_report"]}

Initial Payment report:
{state["payment_report"]}

Initial Shipping report:
{state["shipping_report"]}

Targeted follow-up evidence:
{followup_text}

Return a concise final diagnosis with:
- root cause
- responsible component or boundary
- supporting evidence
- confidence

Use only the reports above. Do not invent information."""
    return {"final_diagnosis": _model().invoke(prompt).content}
