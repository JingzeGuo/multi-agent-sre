"""LLM-backed service investigators and coordinator."""

import os
from typing import TypedDict

from langchain_openai import ChatOpenAI

from tools import (
    get_checkout_evidence,
    get_payment_evidence,
    get_shipping_evidence,
)

MODEL_CONFIG = {
    "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com",
    "temperature": 0,
}


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


def _investigate(service: str, incident: str, evidence: str) -> str:
    prompt = f"""You are the SRE agent responsible only for {service}.

Incident: {incident}

Local {service} evidence:
{evidence}

Write a concise report with:
- service health status
- observed anomaly
- relevant evidence
- suspected dependency, if any
- confidence

Use only the supplied local evidence. Do not invent information."""
    return _model().invoke(prompt).content


def checkout_agent(state: SREState) -> dict[str, str]:
    report = _investigate(
        "Checkout",
        state["incident"],
        get_checkout_evidence(),
    )
    return {"checkout_report": report}


def payment_agent(state: SREState) -> dict[str, str]:
    report = _investigate(
        "Payment",
        state["incident"],
        get_payment_evidence(),
    )
    return {"payment_report": report}


def shipping_agent(state: SREState) -> dict[str, str]:
    report = _investigate(
        "Shipping",
        state["incident"],
        get_shipping_evidence(),
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
