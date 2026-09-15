"""LLM-backed routing, service investigation, and incident coordination."""

import operator
import os
import threading
from typing import Annotated, Any, Literal, TypedDict, cast

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.types import Send

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
MAX_INITIAL_AGENTS = 2

AgentName = Literal["checkout", "payment", "shipping"]
AGENT_NAMES: tuple[AgentName, ...] = ("checkout", "payment", "shipping")

SERVICE_DESCRIPTIONS: dict[AgentName, str] = {
    "checkout": "handles cart checkout and order placement; calls Payment and Shipping",
    "payment": "handles payment authorization and card charging",
    "shipping": "handles shipping quotes, delivery options, and shipment creation",
}


class RouterDecision(TypedDict):
    """Structured output produced by the telemetry-blind router."""

    selected_agents: list[AgentName]
    reason: str


class FollowupDecision(TypedDict):
    """Coordinator decision after reading the available specialist reports."""

    next_agent: Literal["checkout", "payment", "shipping", "none"]
    reason: str


class PeerMessage(TypedDict):
    sender: AgentName
    recipient: AgentName
    question: str
    response: str


def _merge_unique_agents(
    current: list[AgentName],
    update: list[AgentName],
) -> list[AgentName]:
    return list(dict.fromkeys([*current, *update]))


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
    investigation_agent: AgentName
    selected_agents: list[AgentName]
    routing_reason: str
    router_fallback: bool
    checkout_report: str | None
    payment_report: str | None
    shipping_report: str | None
    peer_budget: PeerBudget
    peer_messages: Annotated[list[PeerMessage], operator.add]
    activated_agents: Annotated[list[AgentName], _merge_unique_agents]
    followup_agent: AgentName | None
    followup_reason: str
    final_diagnosis: str | None


SERVICE_CONFIG: dict[AgentName, tuple[str, list[BaseTool]]] = {
    "checkout": ("Checkout", CHECKOUT_TOOLS),
    "payment": ("Payment", PAYMENT_TOOLS),
    "shipping": ("Shipping", SHIPPING_TOOLS),
}


def _model(*, thinking: bool | None = None) -> ChatOpenAI:
    config: dict[str, Any] = {
        **MODEL_CONFIG,
        "api_key": os.environ["DEEPSEEK_API_KEY"],
    }
    if thinking is not None:
        config["extra_body"] = {
            "thinking": {"type": "enabled" if thinking else "disabled"}
        }
    return ChatOpenAI(**config)


def _unique_agents(agents: object, limit: int | None = None) -> list[AgentName]:
    if not isinstance(agents, list):
        return []

    selected: list[AgentName] = []
    for agent in agents:
        if agent in AGENT_NAMES and agent not in selected:
            selected.append(cast(AgentName, agent))
        if limit is not None and len(selected) == limit:
            break
    return selected


def router_agent(state: SREState) -> dict[str, object]:
    """Select at most two initial specialists without access to telemetry."""
    topology = "\n".join(
        f"- {agent}: {description}"
        for agent, description in SERVICE_DESCRIPTIONS.items()
    )
    messages = [
        SystemMessage(
            content=f"""You are an incident triage router, not a diagnostician.
Choose the smallest useful set of one or two service agents to investigate first.
Use only the incident description and static service topology below. You have no
access to logs, traces, metrics, health status, or previous investigation results.
Do not claim a root cause. Return selected_agents and a brief routing reason.

Service topology:
{topology}"""
        ),
        HumanMessage(content=f"Incident: {state['incident']}"),
    ]

    try:
        raw_decision = _model(thinking=False).with_structured_output(
            RouterDecision,
            method="function_calling",
        ).invoke(messages)
        selected = _unique_agents(
            raw_decision.get("selected_agents"),
            MAX_INITIAL_AGENTS,
        )
        reason = str(raw_decision.get("reason", "")).strip()
        if not selected or not reason:
            raise ValueError("router returned an empty or invalid decision")
        return {
            "selected_agents": selected,
            "routing_reason": reason,
            "router_fallback": False,
        }
    except Exception as error:
        # Availability is more important than cost when triage itself is unavailable.
        print(f"\n[Router] Falling back to all agents: {error}")
        return {
            "selected_agents": list(AGENT_NAMES),
            "routing_reason": (
                "Router failed, so all service agents were activated as a safe fallback."
            ),
            "router_fallback": True,
        }


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
        """Activate another service agent for one targeted, non-recursive question."""
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
once to activate the responsible peer for a targeted check. Do not ask unless its
answer could change or materially strengthen your diagnosis. Incorporate any peer
evidence in your report. Do not invent information.

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


def dispatch_selected(state: SREState) -> list[Send]:
    """Fan out graph tasks only for the specialists selected by the router."""
    selected = _unique_agents(state.get("selected_agents")) or list(AGENT_NAMES)
    return [
        Send(
            "service_investigator",
            {
                "incident": state["incident"],
                "investigation_agent": agent,
                "peer_budget": state["peer_budget"],
            },
        )
        for agent in selected
    ]


def service_investigator(state: SREState) -> dict[str, object]:
    """Run one dynamically selected service specialist."""
    agent = state["investigation_agent"]
    report, peer_messages = _investigate(
        agent,
        state["incident"],
        state["peer_budget"],
    )
    activated = [agent]
    for message in peer_messages:
        if message["recipient"] not in activated:
            activated.append(message["recipient"])

    return {
        f"{agent}_report": report,
        "peer_messages": peer_messages,
        "activated_agents": activated,
    }


def _report_summary(state: SREState) -> str:
    sections = []
    for agent in AGENT_NAMES:
        report = state.get(f"{agent}_report")
        sections.append(
            f"{agent.title()} report:\n{report or '[not activated / no report]'}"
        )
    return "\n\n".join(sections)


def coordinator_assess(state: SREState) -> dict[str, object]:
    """Request at most one full follow-up from a still-unseen service domain."""
    activated = _unique_agents(state.get("activated_agents"))
    remaining = [agent for agent in AGENT_NAMES if not state.get(f"{agent}_report")]
    if not remaining:
        return {
            "followup_agent": None,
            "followup_reason": "All relevant service domains have already been reached.",
        }

    prompt = f"""You are an incident coordinator assessing investigation coverage.
You may request one full follow-up investigation only if the current evidence points
to a specific uninvestigated service and that report could materially change the RCA.
Do not request broader investigation merely for completeness. You can only read the
specialist reports and peer messages below; you have no raw telemetry access.

Incident: {state['incident']}
Initially selected: {', '.join(state['selected_agents'])}
Already activated (including peer checks): {', '.join(activated)}
Eligible follow-up agents: {', '.join(remaining)}

{_report_summary(state)}

Peer messages:
{state.get('peer_messages') or '[none]'}

Return next_agent as one eligible service name, or "none", plus a brief reason."""

    try:
        decision = _model(thinking=False).with_structured_output(
            FollowupDecision,
            method="function_calling",
        ).invoke(prompt)
        next_agent = decision.get("next_agent", "none")
        reason = str(decision.get("reason", "")).strip()
        if next_agent in remaining:
            return {
                "followup_agent": cast(AgentName, next_agent),
                "followup_reason": reason or "Coordinator requested more evidence.",
            }
        return {
            "followup_agent": None,
            "followup_reason": reason or "Current reports are sufficient for synthesis.",
        }
    except Exception as error:
        print(f"\n[Coordinator] Coverage assessment failed: {error}")
        return {
            "followup_agent": None,
            "followup_reason": (
                "Coverage assessment failed; continuing with the available evidence."
            ),
        }


def route_after_assessment(state: SREState) -> Literal["followup", "final"]:
    return "followup" if state.get("followup_agent") else "final"


def targeted_followup(state: SREState) -> dict[str, object]:
    agent = state.get("followup_agent")
    if agent is None:
        return {}

    report, peer_messages = _investigate(
        agent,
        state["incident"],
        state["peer_budget"],
    )
    activated = [agent, *(message["recipient"] for message in peer_messages)]

    return {
        f"{agent}_report": report,
        "peer_messages": peer_messages,
        "activated_agents": activated,
    }


def final_coordinator(state: SREState) -> dict[str, str]:
    prompt = f"""You are the incident coordinator. Synthesize the available service
reports into a final diagnosis. A missing report means that service was not activated;
it is not evidence that the service is healthy.

Incident: {state['incident']}
Initially selected agents: {', '.join(state['selected_agents'])}
Routing reason: {state['routing_reason']}
Actually activated agents: {', '.join(state['activated_agents'])}
Coverage assessment: {state['followup_reason']}

{_report_summary(state)}

Peer messages:
{state.get('peer_messages') or '[none]'}

Return a concise final diagnosis with:
- root cause
- responsible component or boundary
- supporting evidence
- confidence

Use only the reports and peer evidence above. Do not invent information. Explicitly
state when the evidence is insufficient instead of treating an uninvestigated service
as healthy."""
    return {"final_diagnosis": _model().invoke(prompt).content}
