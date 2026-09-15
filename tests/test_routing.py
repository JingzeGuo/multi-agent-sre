"""Unit tests for selective routing and on-demand agent expansion."""

import unittest
from unittest.mock import patch

import agents
import app
from agents import PeerBudget
from app import build_graph


class _StructuredModel:
    def __init__(self, result):
        self.result = result
        self.invocations = []

    def invoke(self, value):
        self.invocations.append(value)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _FakeModel:
    def __init__(self, result):
        self.structured = _StructuredModel(result)

    def with_structured_output(self, _schema, **_kwargs):
        return self.structured


class RoutingTests(unittest.TestCase):
    def test_router_selects_at_most_two_known_agents(self):
        model = _FakeModel(
            {
                "selected_agents": [
                    "shipping",
                    "shipping",
                    "checkout",
                    "payment",
                ],
                "reason": "Shipping latency may involve checkout orchestration.",
            }
        )

        with patch.object(agents, "_model", return_value=model) as model_factory:
            result = agents.router_agent(
                {"incident": "International shipping quotes are slow."}
            )

        model_factory.assert_called_once_with(thinking=False)
        self.assertEqual(result["selected_agents"], ["shipping", "checkout"])
        self.assertFalse(result["router_fallback"])
        prompt = "\n".join(
            str(message.content) for message in model.structured.invocations[0]
        )
        self.assertIn("International shipping quotes are slow.", prompt)
        self.assertIn("static service topology", prompt)
        self.assertNotIn("Checkout report", prompt)

    def test_router_failure_falls_back_to_all_agents(self):
        model = _FakeModel(RuntimeError("router unavailable"))

        with patch.object(agents, "_model", return_value=model):
            result = agents.router_agent({"incident": "Checkout is failing."})

        self.assertEqual(result["selected_agents"], list(agents.AGENT_NAMES))
        self.assertTrue(result["router_fallback"])

    def test_router_dispatches_only_selected_agents(self):
        state = {
            "incident": "International shipping quotes are slow.",
            "selected_agents": ["shipping"],
            "peer_budget": PeerBudget(),
        }

        sends = agents.dispatch_selected(state)

        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "service_investigator")
        self.assertEqual(sends[0].arg["investigation_agent"], "shipping")

    def test_peer_activation_is_recorded_by_dynamic_investigator(self):
        peer_message = {
            "sender": "shipping",
            "recipient": "checkout",
            "question": "Are calls to Shipping failing?",
            "response": "Checkout sees elevated Shipping latency.",
        }

        def fake_investigate(agent, _incident, _budget):
            self.assertEqual(agent, "shipping")
            return "Shipping report", [peer_message]

        state = {
            "incident": "International shipping quotes are slow.",
            "investigation_agent": "shipping",
            "peer_budget": PeerBudget(),
        }
        with patch.object(agents, "_investigate", side_effect=fake_investigate):
            result = agents.service_investigator(state)

        self.assertEqual(result["shipping_report"], "Shipping report")
        self.assertNotIn("checkout_report", result)
        self.assertNotIn("payment_report", result)
        self.assertEqual(result["activated_agents"], ["shipping", "checkout"])

    def test_coordinator_can_promote_peer_check_to_full_followup(self):
        model = _FakeModel(
            {
                "next_agent": "checkout",
                "reason": "The Shipping report points to the caller boundary.",
            }
        )
        state = {
            "incident": "International shipping quotes are slow.",
            "selected_agents": ["shipping"],
            "routing_reason": "Shipping owns quote generation.",
            "shipping_report": "Local work is fast; inbound calls are delayed.",
            "peer_messages": [
                {
                    "sender": "shipping",
                    "recipient": "checkout",
                    "question": "Do you see delayed Shipping calls?",
                    "response": "Yes, delays appear at the Checkout boundary.",
                }
            ],
            "activated_agents": ["shipping", "checkout"],
        }

        with patch.object(agents, "_model", return_value=model) as model_factory:
            result = agents.coordinator_assess(state)

        model_factory.assert_called_once_with(thinking=False)
        self.assertEqual(result["followup_agent"], "checkout")

    def test_graph_contains_optional_followup_path(self):
        mermaid = build_graph().get_graph().draw_mermaid()
        self.assertIn("service_investigator", mermaid)
        self.assertIn("coordinator_assess", mermaid)
        self.assertIn("targeted_followup", mermaid)

    def test_graph_reduces_dynamic_fanout_before_coordinator(self):
        coordinator_calls = []

        def fake_router(_state):
            return {
                "selected_agents": ["checkout", "payment"],
                "routing_reason": "Both domains are relevant.",
                "router_fallback": False,
            }

        def fake_investigator(state):
            agent = state["investigation_agent"]
            return {
                f"{agent}_report": f"{agent} report",
                "peer_messages": [],
                "activated_agents": [agent],
            }

        def fake_assess(state):
            coordinator_calls.append(state)
            self.assertEqual(state["checkout_report"], "checkout report")
            self.assertEqual(state["payment_report"], "payment report")
            return {
                "followup_agent": None,
                "followup_reason": "Evidence is sufficient.",
            }

        def fake_final(_state):
            return {"final_diagnosis": "Payment is the root cause."}

        with (
            patch.object(app, "router_agent", fake_router),
            patch.object(app, "service_investigator", fake_investigator),
            patch.object(app, "coordinator_assess", fake_assess),
            patch.object(app, "final_coordinator", fake_final),
        ):
            result = app.build_graph().invoke(
                {
                    "incident": "Users cannot complete checkout.",
                    "peer_budget": PeerBudget(),
                    "peer_messages": [],
                }
            )

        self.assertEqual(len(coordinator_calls), 1)
        self.assertEqual(result["activated_agents"], ["checkout", "payment"])


if __name__ == "__main__":
    unittest.main()
