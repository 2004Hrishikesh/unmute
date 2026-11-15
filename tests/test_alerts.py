import asyncio

import server.alerts as alerts

URGENT = ["I cannot breathe", "I am in pain", "Help me"]


def test_is_urgent_exact_match_only():
    assert alerts.is_urgent("I cannot breathe", URGENT)
    assert alerts.is_urgent("  i CANNOT breathe ", URGENT)  # case and whitespace do not matter
    assert not alerts.is_urgent("I cannot breathe now", URGENT)  # no substring matching
    assert not alerts.is_urgent("Yes", URGENT)
    assert not alerts.is_urgent("", URGENT)
    assert not alerts.is_urgent("Help me", [])


def test_push_ntfy_without_topic_does_not_raise():
    # Not monkeypatched here on purpose: with no topic it must return before any network call.
    ok, detail = asyncio.run(alerts.push_ntfy.__wrapped__("", "x", "Bed 1")) if hasattr(alerts.push_ntfy, "__wrapped__") \
        else asyncio.run(_real_push("", "x", "Bed 1"))
    assert ok is False and "topic" in detail


async def _real_push(topic, sentence, bed):
    # conftest replaces alerts.push_ntfy; reach the original through the module source to keep this
    # test honest about the no-topic early return.
    import importlib
    import server.alerts as fresh
    importlib.reload(fresh)
    try:
        return await fresh.push_ntfy(topic, sentence, bed)
    finally:
        importlib.reload(fresh)
