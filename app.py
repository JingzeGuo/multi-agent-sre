"""Minimal parallel multi-agent SRE workflow."""

from langgraph.graph import END, START, StateGraph

from agents import (
    SREState,
    checkout_agent,
    coordinator,
    payment_agent,
    shipping_agent,
)


def build_graph():
    builder = StateGraph(SREState)

    builder.add_node("checkout_agent", checkout_agent)
    builder.add_node("payment_agent", payment_agent)
    builder.add_node("shipping_agent", shipping_agent)
    builder.add_node("coordinator", coordinator)

    builder.add_edge(START, "checkout_agent")
    builder.add_edge(START, "payment_agent")
    builder.add_edge(START, "shipping_agent")

    builder.add_edge(
        ["checkout_agent", "payment_agent", "shipping_agent"],
        "coordinator",
    )
    builder.add_edge("coordinator", END)

    return builder.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke({"incident": "Users cannot complete checkout."})

    print("\n=== Checkout Agent ===")
    print(result["checkout_report"])
    print("\n=== Payment Agent ===")
    print(result["payment_report"])
    print("\n=== Shipping Agent ===")
    print(result["shipping_report"])
    print("\n=== Final Diagnosis ===")
    print(result["final_diagnosis"])


if __name__ == "__main__":
    main()
