"""Adaptive multi-agent SRE workflow with selective specialist activation."""

from langgraph.graph import END, START, StateGraph

from agents import (
    AGENT_NAMES,
    PeerBudget,
    SREState,
    coordinator_assess,
    dispatch_selected,
    final_coordinator,
    route_after_assessment,
    router_agent,
    service_investigator,
    targeted_followup,
)


def build_graph():
    builder = StateGraph(SREState)

    builder.add_node("router", router_agent)
    builder.add_node("service_investigator", service_investigator)
    builder.add_node("coordinator_assess", coordinator_assess)
    builder.add_node("targeted_followup", targeted_followup)
    builder.add_node("final_coordinator", final_coordinator)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        dispatch_selected,
        ["service_investigator"],
    )
    builder.add_edge("service_investigator", "coordinator_assess")
    builder.add_conditional_edges(
        "coordinator_assess",
        route_after_assessment,
        {
            "followup": "targeted_followup",
            "final": "final_coordinator",
        },
    )
    builder.add_edge("targeted_followup", "final_coordinator")
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

    print("\n=== Router ===")
    print(f"Selected: {', '.join(result['selected_agents'])}")
    print(f"Reason: {result['routing_reason']}")
    if result["router_fallback"]:
        print("Mode: all-agent safety fallback")

    for agent in AGENT_NAMES:
        report = result.get(f"{agent}_report")
        if report is not None:
            print(f"\n=== {agent.title()} Agent ===")
            print(report)

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

    print("\n=== Activation Summary ===")
    print(f"Actually activated: {', '.join(result['activated_agents'])}")
    print(f"Coverage assessment: {result['followup_reason']}")

    print("\n=== Final Diagnosis ===")
    print(result["final_diagnosis"])


if __name__ == "__main__":
    main()
