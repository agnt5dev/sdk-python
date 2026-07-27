"""Event durability classification tests."""

from agnt5.events import is_checkpoint_event, is_sse_only_event


def test_lm_stream_families_are_transient_events():
    for event_type in (
        "lm.content_block.started",
        "lm.content_block.delta",
        "lm.content_block.completed",
        "lm.message.delta",
        "lm.thinking.delta",
        "lm.tool_call.start",
        "lm.tool_call.delta",
        "lm.tool_call.stop",
    ):
        assert is_sse_only_event(event_type)
        assert not is_checkpoint_event(event_type)
