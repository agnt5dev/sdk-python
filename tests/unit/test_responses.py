"""Unit tests for response classes."""

from datetime import datetime, timedelta, timezone

import pytest

from agnt5.responses import (
    EventsResponse,
    Event,
    RunErrorDetail,
    RunResponse,
    RunStatus,
    StatusResponse,
    SubmitLinks,
    SubmitResponse,
    parse_events_response,
    parse_run_response,
    parse_status_response,
    parse_submit_response,
)


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_status_values(self):
        """Verify all expected status values exist."""
        assert RunStatus.ENQUEUED == "enqueued"
        assert RunStatus.STARTED == "started"
        assert RunStatus.RUNNING == "running"
        assert RunStatus.COMPLETED == "completed"
        assert RunStatus.FAILED == "failed"
        assert RunStatus.CANCELLED == "cancelled"
        assert RunStatus.PAUSED == "paused"
        assert RunStatus.AWAITING_INPUT == "awaiting_input"
        assert RunStatus.TIMEOUT == "timeout"
        assert RunStatus.UNKNOWN == "unknown"


class TestRunErrorDetail:
    """Tests for RunErrorDetail dataclass."""

    def test_basic_error(self):
        """Test creating a basic error."""
        error = RunErrorDetail(code="TEST_ERROR", message="Test message")
        assert error.code == "TEST_ERROR"
        assert error.message == "Test message"
        assert error.details is None

    def test_error_with_details(self):
        """Test creating an error with details."""
        error = RunErrorDetail(
            code="TEST_ERROR",
            message="Test message",
            details={"key": "value"},
        )
        assert error.details == {"key": "value"}

    def test_str_without_details(self):
        """Test string representation without details."""
        error = RunErrorDetail(code="ERR", message="Something went wrong")
        assert str(error) == "ERR: Something went wrong"

    def test_str_with_details(self):
        """Test string representation with details."""
        error = RunErrorDetail(
            code="ERR",
            message="Something went wrong",
            details={"file": "test.py"},
        )
        assert "ERR: Something went wrong" in str(error)
        assert "details" in str(error)


class TestRunResponse:
    """Tests for RunResponse dataclass."""

    def test_successful_response(self):
        """Test a successful response."""
        response = RunResponse(
            run_id="test-123",
            status_code=200,
            status=RunStatus.COMPLETED,
            output={"result": "success"},
            duration_ms=100,
        )
        assert response.run_id == "test-123"
        assert response.is_success
        assert not response.is_error
        assert not response.is_pending
        assert response.output == {"result": "success"}

    def test_pending_response(self):
        """Test a pending response."""
        response = RunResponse(
            run_id="test-123",
            status_code=202,
            status=RunStatus.RUNNING,
        )
        assert response.is_pending
        assert not response.is_success
        assert not response.is_error

    def test_failed_response(self):
        """Test a failed response."""
        response = RunResponse(
            run_id="test-123",
            status_code=500,
            status=RunStatus.FAILED,
            error=RunErrorDetail(code="EXEC_FAILED", message="Execution failed"),
        )
        assert response.is_error
        assert not response.is_success
        assert not response.is_pending

    def test_elapsed_property(self):
        """Test elapsed timedelta conversion."""
        response = RunResponse(
            run_id="test-123",
            status_code=200,
            status=RunStatus.COMPLETED,
            duration_ms=1500,
        )
        assert response.elapsed == timedelta(milliseconds=1500)
        assert response.elapsed.total_seconds() == 1.5

    def test_elapsed_none_when_no_duration(self):
        """Test elapsed is None when no duration."""
        response = RunResponse(
            run_id="test-123",
            status_code=200,
            status=RunStatus.COMPLETED,
        )
        assert response.elapsed is None

    def test_raise_for_status_success(self):
        """Test raise_for_status does nothing on success."""
        response = RunResponse(
            run_id="test-123",
            status_code=200,
            status=RunStatus.COMPLETED,
        )
        # Should not raise
        response.raise_for_status()

    def test_raise_for_status_error(self):
        """Test raise_for_status raises on error."""
        from agnt5.client import RunError

        response = RunResponse(
            run_id="test-123",
            status_code=500,
            status=RunStatus.FAILED,
            error=RunErrorDetail(code="TEST_ERR", message="Test error"),
        )
        with pytest.raises(RunError) as exc_info:
            response.raise_for_status()
        assert exc_info.value.run_id == "test-123"
        assert exc_info.value.error_code == "TEST_ERR"


class TestSubmitResponse:
    """Tests for SubmitResponse dataclass."""

    def test_basic_submit_response(self):
        """Test basic submit response."""
        response = SubmitResponse(
            run_id="test-456",
            status_code=202,
            status=RunStatus.ENQUEUED,
            trace_id="trace-abc",
        )
        assert response.run_id == "test-456"
        assert response.status_code == 202
        assert response.status == RunStatus.ENQUEUED

    def test_status_url_property(self):
        """Test status_url property."""
        response = SubmitResponse(
            run_id="test-456",
            status_code=202,
            status=RunStatus.ENQUEUED,
            links=SubmitLinks(self_url="/v1/runs/test-456"),
        )
        assert response.status_url == "/v1/runs/test-456"

    def test_status_url_none_without_links(self):
        """Test status_url is None without links."""
        response = SubmitResponse(
            run_id="test-456",
            status_code=202,
            status=RunStatus.ENQUEUED,
        )
        assert response.status_url is None


class TestStatusResponse:
    """Tests for StatusResponse dataclass."""

    def test_is_complete_for_completed(self):
        """Test is_complete for completed status."""
        response = StatusResponse(
            run_id="test-789",
            status_code=200,
            status=RunStatus.COMPLETED,
        )
        assert response.is_complete
        assert not response.is_running

    def test_is_complete_for_failed(self):
        """Test is_complete for failed status."""
        response = StatusResponse(
            run_id="test-789",
            status_code=500,
            status=RunStatus.FAILED,
        )
        assert response.is_complete
        assert not response.is_running

    def test_is_running_for_running(self):
        """Test is_running for running status."""
        response = StatusResponse(
            run_id="test-789",
            status_code=202,
            status=RunStatus.RUNNING,
        )
        assert response.is_running
        assert not response.is_complete


class TestParseRunResponse:
    """Tests for parse_run_response function."""

    def test_parse_successful_response(self):
        """Test parsing a successful response."""
        data = {
            "run_id": "abc-123",
            "status_code": 200,
            "status": "completed",
            "output": {"message": "Hello!"},
            "duration_ms": 50,
            "trace_id": "trace-xyz",
            "function": "greet",
            "created_at": "2026-01-16T19:23:37.433905Z",
            "completed_at": "2026-01-16T19:23:37.450628Z",
        }
        response = parse_run_response(data)

        assert response.run_id == "abc-123"
        assert response.status_code == 200
        assert response.status == RunStatus.COMPLETED
        assert response.output == {"message": "Hello!"}
        assert response.duration_ms == 50
        assert response.trace_id == "trace-xyz"
        assert response.component == "greet"
        assert response.created_at is not None
        assert response.completed_at is not None

    def test_parse_error_response(self):
        """Test parsing an error response."""
        data = {
            "run_id": "abc-123",
            "status_code": 500,
            "status": "failed",
            "error": {
                "code": "EXECUTION_FAILED",
                "message": "Something went wrong",
            },
        }
        response = parse_run_response(data)

        assert response.is_error
        assert response.error is not None
        assert response.error.code == "EXECUTION_FAILED"
        assert response.error.message == "Something went wrong"

    def test_parse_string_error(self):
        """Test parsing a string error (legacy format)."""
        data = {
            "run_id": "abc-123",
            "status_code": 500,
            "status": "failed",
            "error": "Legacy error message",
        }
        response = parse_run_response(data)

        assert response.error is not None
        assert response.error.code == "UNKNOWN"
        assert response.error.message == "Legacy error message"

    def test_parse_nested_output(self):
        """Test parsing nested output structure."""
        data = {
            "run_id": "abc-123",
            "status_code": 200,
            "status": "completed",
            "result": {
                "output": {
                    "output_data": "Nested output",
                },
            },
        }
        response = parse_run_response(data)
        assert response.output == "Nested output"

    def test_parse_unknown_status(self):
        """Test parsing unknown status value."""
        data = {
            "run_id": "abc-123",
            "status_code": 200,
            "status": "some_new_status",
        }
        response = parse_run_response(data)
        assert response.status == RunStatus.UNKNOWN

    def test_inferred_status_code_from_status(self):
        """Test status code inference from status."""
        data = {
            "run_id": "abc-123",
            "status": "completed",
        }
        response = parse_run_response(data)
        assert response.status_code == 200

        data = {
            "run_id": "abc-123",
            "status": "running",
        }
        response = parse_run_response(data)
        assert response.status_code == 202


class TestParseSubmitResponse:
    """Tests for parse_submit_response function."""

    def test_parse_submit_response(self):
        """Test parsing a submit response."""
        data = {
            "run_id": "def-456",
            "status_code": 202,
            "status": "enqueued",
            "trace_id": "trace-123",
            "function": "process",
            "created_at": "2026-01-16T19:23:33.208952Z",
            "links": {
                "self": "/v1/runs/def-456",
            },
        }
        response = parse_submit_response(data)

        assert response.run_id == "def-456"
        assert response.status_code == 202
        assert response.status == RunStatus.ENQUEUED
        assert response.trace_id == "trace-123"
        assert response.component == "process"
        assert response.links is not None
        assert response.status_url == "/v1/runs/def-456"


class TestParseStatusResponse:
    """Tests for parse_status_response function."""

    def test_parse_status_response_snake_case(self):
        """Test parsing status response with snake_case fields."""
        data = {
            "run_id": "ghi-789",
            "status_code": 202,
            "status": "running",
            "created_at": "2026-01-16T19:23:33Z",
            "started_at": "2026-01-16T19:23:34Z",
        }
        response = parse_status_response(data)

        assert response.run_id == "ghi-789"
        assert response.status == RunStatus.RUNNING
        assert response.started_at is not None

    def test_parse_status_response_camel_case(self):
        """Test parsing status response with camelCase fields (legacy)."""
        data = {
            "runId": "ghi-789",
            "status": "completed",
            "submittedAt": "2026-01-16T19:23:33Z",
            "startedAt": "2026-01-16T19:23:34Z",
            "completedAt": "2026-01-16T19:23:35Z",
        }
        response = parse_status_response(data)

        assert response.run_id == "ghi-789"
        assert response.status == RunStatus.COMPLETED
        assert response.created_at is not None
        assert response.started_at is not None
        assert response.completed_at is not None


class TestEvent:
    """Tests for Event dataclass."""

    def test_basic_event(self):
        """Test creating a basic journal event."""
        event = Event(
            id="evt-123",
            event_type="run.started",
            run_id="run-456",
        )
        assert event.id == "evt-123"
        assert event.event_type == "run.started"
        assert event.run_id == "run-456"
        assert event.step_key is None
        assert event.input_data is None
        assert event.output_data is None

    def test_event_with_all_fields(self):
        """Test creating an event with all fields."""
        event = Event(
            id="evt-123",
            event_type="lm.content_block.delta",
            run_id="run-456",
            step_key="step-1",
            input_data={"prompt": "Hello"},
            output_data={"content": "Hi there"},
            parent_event_id="evt-100",
            metadata={"model": "gpt-4"},
            trace_id="trace-abc",
            timestamp_ns=1705427017000000000,
            created_at=datetime(2026, 1, 16, 19, 23, 37),
        )
        assert event.step_key == "step-1"
        assert event.input_data == {"prompt": "Hello"}
        assert event.output_data == {"content": "Hi there"}
        assert event.parent_event_id == "evt-100"
        assert event.metadata == {"model": "gpt-4"}
        assert event.trace_id == "trace-abc"
        assert event.timestamp_ns == 1705427017000000000
        assert event.created_at is not None


class TestEventsResponse:
    """Tests for EventsResponse dataclass."""

    def test_empty_events_response(self):
        """Test creating an empty events response."""
        response = EventsResponse(items=[], count=0)
        assert len(response) == 0
        assert list(response) == []

    def test_events_response_with_items(self):
        """Test events response with multiple items."""
        events = [
            Event(id="1", event_type="run.started", run_id="run-1"),
            Event(id="2", event_type="run.completed", run_id="run-1"),
        ]
        response = EventsResponse(items=events, count=2)

        assert len(response) == 2
        assert response.count == 2

        # Test iteration
        event_ids = [e.id for e in response]
        assert event_ids == ["1", "2"]

    def test_events_response_iteration(self):
        """Test that EventsResponse supports iteration."""
        events = [
            Event(id=f"evt-{i}", event_type="test", run_id="run-1")
            for i in range(3)
        ]
        response = EventsResponse(items=events, count=3)

        # Direct iteration
        count = 0
        for event in response:
            assert event.event_type == "test"
            count += 1
        assert count == 3


class TestParseEventsResponse:
    """Tests for parse_events_response function."""

    def test_parse_empty_response(self):
        """Test parsing an empty events response."""
        data = {"items": [], "count": 0}
        response = parse_events_response(data)

        assert len(response) == 0
        assert response.count == 0

    def test_parse_events_response(self):
        """Test parsing a full events response."""
        data = {
            "items": [
                {
                    "id": "evt-001",
                    "event_type": "run.started",
                    "run_id": "run-abc",
                    "created_at": "2026-01-16T19:23:33Z",
                },
                {
                    "id": "evt-002",
                    "event_type": "lm.content_block.delta",
                    "run_id": "run-abc",
                    "output_data": {"content": "Hello"},
                    "created_at": "2026-01-16T19:23:34Z",
                },
                {
                    "id": "evt-003",
                    "event_type": "run.completed",
                    "run_id": "run-abc",
                    "created_at": "2026-01-16T19:23:35Z",
                },
            ],
            "count": 3,
        }
        response = parse_events_response(data)

        assert response.count == 3
        assert len(response.items) == 3

        # Check first event
        assert response.items[0].id == "evt-001"
        assert response.items[0].event_type == "run.started"
        assert response.items[0].run_id == "run-abc"
        assert response.items[0].created_at is not None

        # Check second event with output_data
        assert response.items[1].id == "evt-002"
        assert response.items[1].event_type == "lm.content_block.delta"
        assert response.items[1].output_data == {"content": "Hello"}

    def test_parse_event_with_all_fields(self):
        """Test parsing an event with all fields populated."""
        data = {
            "items": [
                {
                    "id": "evt-full",
                    "event_type": "function.called",
                    "run_id": "run-xyz",
                    "step_key": "step-1",
                    "input_data": {"arg": "value"},
                    "output_data": {"result": "success"},
                    "parent_event_id": "evt-parent",
                    "metadata": {"key": "value"},
                    "trace_id": "trace-123",
                    "timestamp_ns": 1705427017000000000,
                    "created_at": "2026-01-16T19:23:37.433905Z",
                },
            ],
            "count": 1,
        }
        response = parse_events_response(data)

        event = response.items[0]
        assert event.id == "evt-full"
        assert event.event_type == "function.called"
        assert event.run_id == "run-xyz"
        assert event.step_key == "step-1"
        assert event.input_data == {"arg": "value"}
        assert event.output_data == {"result": "success"}
        assert event.parent_event_id == "evt-parent"
        assert event.metadata == {"key": "value"}
        assert event.trace_id == "trace-123"
        assert event.timestamp_ns == 1705427017000000000
        assert event.created_at is not None

    def test_parse_events_infers_count(self):
        """Test that count is inferred if not provided."""
        data = {
            "items": [
                {"id": "1", "event_type": "test", "run_id": "run-1"},
                {"id": "2", "event_type": "test", "run_id": "run-1"},
            ],
        }
        response = parse_events_response(data)

        # Count should be inferred from items length
        assert response.count == 2
