"""LLM-backed service investigators and coordinator."""

import operator
import os
import threading
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

from tools import CHECKOUT_TOOLS, PAYMENT_TOOLS, SHIPPING_TOOLS

load_dotenv()

MODEL_CONFIG = {
    "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com",
    "temperature": 0,
}

INITIAL_TOOL_CALLS = 5
PEER_TOOL_CALLS = 2
MAX_PEER_MESSAGES = 2

AgentName = Literal["checkout", "payment", "shipping"]


class PeerMessage(TypedDict):
    sender: AgentName
    recipient: AgentName
    question: str
    response: str


class PeerBudget:
    """Per-run, thread-safe communication limits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._senders: set[AgentName] = set()
        self._total = 0

    def reserve(self, sender: AgentName) -> str | None:
        with self._lock:
            if sender in self._senders:
                return "You have already used your one peer question."
            if self._total >= MAX_PEER_MESSAGES:
                return "The global peer-message limit has been reached."
            self._senders.add(sender)
            self._total += 1
        return None


class SREState(TypedDict):
    incident: str
    checkout_report: str | None
    payment_report: str | None
    shipping_report: str | None
    peer_budget: PeerBudget
    peer_messages: Annotated[list[PeerMessage], operator.add]
    final_diagnosis: str | None


SERVICE_CONFIG: dict[AgentName, tuple[str, list[BaseTool]]] = {
    "checkout": ("Checkout", CHECKOUT_TOOLS),
    "payment": ("Payment", PAYMENT_TOOLS),
    "shipping": ("Shipping", SHIPPING_TOOLS),
}


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        **MODEL_CONFIG,
        api_key=os.environ["DEEPSEEK_API_KEY"],
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


def _answer_peer_question(
    recipient: AgentName,
    incident: str,
    question: str,
) -> str:
    service, service_tools = SERVICE_CONFIG[recipient]
    messages = [
        SystemMessage(
            content=f"""You are the {service} SRE agent answering a peer question.
Use only your own {service} tools and at most {PEER_TOOL_CALLS} tool calls.
You cannot ask or forward questions to another agent.
Return a concise answer with evidence and confidence. Do not invent information."""
        ),
        HumanMessage(content=f"Incident: {incident}\nPeer question: {question}"),
    ]
    return _run_tool_loop(
        f"{service} peer",
        messages,
        service_tools,
        PEER_TOOL_CALLS,
    )


def _peer_tool(
    sender: AgentName,
    incident: str,
    budget: PeerBudget,
    peer_messages: list[PeerMessage],
) -> BaseTool:
    @tool
    def ask_agent(agent_name: AgentName, question: str) -> str:
        """Ask one other service agent one targeted, non-recursive question."""
        if agent_name == sender:
            return "Choose another service agent, not yourself."

        denied = budget.reserve(sender)
        if denied:
            return denied

        response = _answer_peer_question(agent_name, incident, question)
        peer_messages.append(
            {
                "sender": sender,
                "recipient": agent_name,
                "question": question,
                "response": response,
            }
        )
        return f"{agent_name.title()} peer response:\n{response}"

    return ask_agent


def _investigate(
    agent: AgentName,
    incident: str,
    budget: PeerBudget,
) -> tuple[str, list[PeerMessage]]:
    service, service_tools = SERVICE_CONFIG[agent]
    peer_messages: list[PeerMessage] = []
    available_tools = [
        *service_tools,
        _peer_tool(agent, incident, budget, peer_messages),
    ]
    messages = [
        SystemMessage(
            content=f"""You are the SRE agent responsible only for {service}.
Investigate independently using only the supplied {service} tools.
Call one tool at a time and make at most {INITIAL_TOOL_CALLS} tool calls.
If local evidence leaves a specific dependency uncertainty, you may use ask_agent
once to ask the responsible peer. Do not ask unless its answer could change or
materially strengthen your diagnosis. Incorporate any peer evidence in your report.
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
    report = _run_tool_loop(
        service,
        messages,
        available_tools,
        INITIAL_TOOL_CALLS,
    )
    return report, peer_messages


def checkout_agent(state: SREState) -> dict[str, object]:
    report, peer_messages = _investigate(
        "checkout", state["incident"], state["peer_budget"]
    )
    return {"checkout_report": report, "peer_messages": peer_messages}


def payment_agent(state: SREState) -> dict[str, object]:
    report, peer_messages = _investigate(
        "payment", state["incident"], state["peer_budget"]
    )
    return {"payment_report": report, "peer_messages": peer_messages}


def shipping_agent(state: SREState) -> dict[str, object]:
    report, peer_messages = _investigate(
        "shipping", state["incident"], state["peer_budget"]
    )
    return {"shipping_report": report, "peer_messages": peer_messages}


def final_coordinator(state: SREState) -> dict[str, str]:
    prompt = f"""You are the incident coordinator. Synthesize the service reports.

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
