"""Minimal parallel multi-agent SRE workflow."""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents import (
    SREState,
    checkout_agent,
    coordinator_assess,
    final_coordinator,
    follow_up,
    payment_agent,
    shipping_agent,
)


def route_after_assessment(state: SREState) -> str | list[Send]:
    if not state["followups"]:
        return "final_coordinator"

    previous_reports = {
        "checkout": state["checkout_report"],
        "payment": state["payment_report"],
        "shipping": state["shipping_report"],
    }
    return [
        Send(
            "follow_up",
            {
                "incident": state["incident"],
                "agent": request["agent"],
                "previous_report": previous_reports[request["agent"]] or "",
                "question": request["question"],
            },
        )
        for request in state["followups"]
    ]


def build_graph():
    builder = StateGraph(SREState)

    builder.add_node("checkout_agent", checkout_agent)
    builder.add_node("payment_agent", payment_agent)
    builder.add_node("shipping_agent", shipping_agent)
    builder.add_node("coordinator_assess", coordinator_assess)
    builder.add_node("follow_up", follow_up)
    builder.add_node("final_coordinator", final_coordinator, defer=True)

    builder.add_edge(START, "checkout_agent")
    builder.add_edge(START, "payment_agent")
    builder.add_edge(START, "shipping_agent")

    builder.add_edge(
        ["checkout_agent", "payment_agent", "shipping_agent"],
        "coordinator_assess",
    )
    builder.add_conditional_edges(
        "coordinator_assess",
        route_after_assessment,
        ["follow_up", "final_coordinator"],
    )
    builder.add_edge("follow_up", "final_coordinator")
    builder.add_edge("final_coordinator", END)

    return builder.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "incident": "Users cannot complete checkout.",
            "followups": [],
            "followup_reports": {},
        }
    )

    print("\n=== Checkout Agent ===")
    print(result["checkout_report"])
    print("\n=== Payment Agent ===")
    print(result["payment_report"])
    print("\n=== Shipping Agent ===")
    print(result["shipping_report"])

    if result["followups"]:
        print("\n=== Coordinator Follow-ups ===")
        for request in result["followups"]:
            print(f"{request['agent'].title()}: {request['question']}")

        print("\n=== Follow-up Reports ===")
        for agent, report in result["followup_reports"].items():
            print(f"\n--- {agent.title()} ---")
            print(report)
    else:
        print("\n=== Coordinator Assessment ===")
        print("Initial evidence was sufficient; no follow-up requested.")

    print("\n=== Final Diagnosis ===")
    print(result["final_diagnosis"])


if __name__ == "__main__":
    main()
