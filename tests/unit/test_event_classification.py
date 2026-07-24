"""Event durability classification tests."""

from agnt5.events import is_checkpoint_event, is_sse_only_event


def test_lm_content_blocks_are_transient_stream_events():
    for event_type in (
        "lm.content_block.started",
        "lm.content_block.delta",
        "lm.content_block.completed",
    ):
        assert is_sse_only_event(event_type)
        assert not is_checkpoint_event(event_type)
