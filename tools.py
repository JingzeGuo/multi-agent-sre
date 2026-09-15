"""Service-scoped stub telemetry tools."""

from langchain_core.tools import tool


def _limit(entries: list[str], requested: int, maximum: int) -> str:
    count = max(0, min(requested, maximum))
    return "\n".join(entries[:count]) or "No matching telemetry found."


def _logs(entries: list[tuple[str, str]], level: str, limit: int) -> str:
    requested_level = level.lower()
    matching = [
        entry
        for entry_level, entry in entries
        if requested_level in {"all", "*"} or entry_level == requested_level
    ]
    return _limit(matching, limit, 50)


@tool
def get_checkout_status() -> str:
    """Return only Checkout's runtime and health-check status."""
    return "container_state=running\nhealth_check=healthy"


@tool
def get_checkout_logs(level: str = "error", limit: int = 50) -> str:
    """Return recent Checkout logs, capped at 50 entries."""
    entries = [
        (
            "info",
            "level=INFO service=CheckoutService message='Checkout service ready'",
        ),
        (
            "error",
            "level=ERROR service=CheckoutService operation=PlaceOrder "
            "message='PlaceOrder failed'",
        ),
    ]
    return _logs(entries, level, limit)


@tool
def get_checkout_traces(limit: int = 20) -> str:
    """Return recent traces visible from Checkout, capped at 20 traces."""
    entries = [
        "trace_id=checkout-7f3 span=CheckoutService/PlaceOrder status=ERROR",
        "trace_id=checkout-7f3 span=PaymentService/Charge status=UNAVAILABLE "
        "error='name resolver error while trying to reach Payment'",
    ]
    return _limit(entries, limit, 20)


@tool
def get_payment_status() -> str:
    """Return only Payment's runtime and health-check status."""
    return "container_state=running\nhealth_check=healthy"


@tool
def get_payment_logs(level: str = "error", limit: int = 50) -> str:
    """Return recent Payment logs, capped at 50 entries."""
    entries = [
        (
            "info",
            "level=INFO service=PaymentService message='Payment service ready'",
        ),
        (
            "info",
            "level=INFO service=PaymentService operation=Charge "
            "message='Charge completed'",
        ),
    ]
    return _logs(entries, level, limit)


@tool
def get_payment_traces(limit: int = 20) -> str:
    """Return recent traces visible from Payment, capped at 20 traces."""
    entries = [
        "window=incident service=PaymentService span=Charge incoming_count=0",
        "trace_id=payment-2a1 span=PaymentService/Charge status=OK",
    ]
    return _limit(entries, limit, 20)


@tool
def get_shipping_status() -> str:
    """Return only Shipping's runtime and health-check status."""
    return "container_state=running\nhealth_check=healthy"


@tool
def get_shipping_logs(level: str = "error", limit: int = 50) -> str:
    """Return recent Shipping logs, capped at 50 entries."""
    entries = [
        (
            "info",
            "level=INFO service=ShippingService message='Shipping service ready'",
        )
    ]
    return _logs(entries, level, limit)


@tool
def get_shipping_traces(limit: int = 20) -> str:
    """Return recent traces visible from Shipping, capped at 20 traces."""
    entries = [
        "trace_id=shipping-9c4 span=ShippingService/GetQuote status=OK",
        "trace_id=shipping-1e8 span=ShippingService/ShipOrder status=OK",
    ]
    return _limit(entries, limit, 20)


CHECKOUT_TOOLS = [
    get_checkout_status,
    get_checkout_logs,
    get_checkout_traces,
]

PAYMENT_TOOLS = [
    get_payment_status,
    get_payment_logs,
    get_payment_traces,
]

SHIPPING_TOOLS = [
    get_shipping_status,
    get_shipping_logs,
    get_shipping_traces,
]
