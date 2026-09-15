"""Stub evidence collectors for the V0.1 incident."""


def get_checkout_evidence() -> str:
    return """- CheckoutService/PlaceOrder failed
- PaymentService/Charge returned gRPC UNAVAILABLE
- A name resolver error occurred while trying to reach Payment"""


def get_payment_evidence() -> str:
    return """- Payment service appears healthy
- No corresponding incoming Charge request was observed"""


def get_shipping_evidence() -> str:
    return """- Shipping service appears healthy
- No relevant error was observed"""
