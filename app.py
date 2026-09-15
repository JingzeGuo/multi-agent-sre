"""Minimal parallel multi-agent SRE workflow."""

from langgraph.graph import END, START, StateGraph

from agents import (
    PeerBudget,
    SREState,
    checkout_agent,
    final_coordinator,
    payment_agent,
    shipping_agent,
)


def build_graph():
    builder = StateGraph(SREState)

    builder.add_node("checkout_agent", checkout_agent)
    builder.add_node("payment_agent", payment_agent)
    builder.add_node("shipping_agent", shipping_agent)
    builder.add_node("final_coordinator", final_coordinator)

    builder.add_edge(START, "checkout_agent")
    builder.add_edge(START, "payment_agent")
    builder.add_edge(START, "shipping_agent")

    builder.add_edge(
        ["checkout_agent", "payment_agent", "shipping_agent"],
        "final_coordinator",
    )
    builder.add_edge("final_coordinator", END)

    return builder.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "incident": "Users cannot complete checkout.",
            "peer_budget": PeerBudget(),
            "peer_messages": [],
        }
    )

    print("\n=== Checkout Agent ===")
    print(result["checkout_report"])
    print("\n=== Payment Agent ===")
    print(result["payment_report"])
    print("\n=== Shipping Agent ===")
    print(result["shipping_report"])

    if result["peer_messages"]:
        print("\n=== Peer Messages ===")
        for message in result["peer_messages"]:
            print(
                f"\n{message['sender'].title()} -> {message['recipient'].title()}"
                f"\nQuestion: {message['question']}"
                f"\nResponse: {message['response']}"
            )
    else:
        print("\n=== Peer Messages ===")
        print("No peer communication was needed.")

    print("\n=== Final Diagnosis ===")
    print(result["final_diagnosis"])


if __name__ == "__main__":
    main()
