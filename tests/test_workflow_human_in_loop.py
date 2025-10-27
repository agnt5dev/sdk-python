"""Unit tests for human-in-the-loop workflow functionality."""

import pytest
from agnt5 import (
    WorkflowContext,
    workflow,
    WaitingForUserInputException,
    with_entity_context,
)
from agnt5.workflow import WorkflowEntity


class TestWaitForUserBasics:
    """Test basic wait_for_user() functionality."""

    @with_entity_context
    async def test_wait_for_user_raises_exception_first_call(self):
        """Test that wait_for_user raises exception on first call."""
        # Create workflow entity and context
        entity = WorkflowEntity(run_id="test-run-123")
        ctx = WorkflowContext(workflow_entity=entity, run_id="test-run-123")

        # First call should raise WaitingForUserInputException
        with pytest.raises(WaitingForUserInputException) as exc_info:
            await ctx.wait_for_user(
                question="Approve this action?",
                input_type="approval",
                options=[
                    {"id": "yes", "label": "Approve"},
                    {"id": "no", "label": "Reject"},
                ],
            )

        # Verify exception contains correct data
        exc = exc_info.value
        assert exc.question == "Approve this action?"
        assert exc.input_type == "approval"
        assert len(exc.options) == 2
        assert exc.options[0]["id"] == "yes"
        assert exc.options[1]["id"] == "no"
        assert isinstance(exc.checkpoint_state, dict)

    @with_entity_context
    async def test_wait_for_user_returns_cached_response(self):
        """Test that wait_for_user returns cached response on replay."""
        # Create workflow entity and context
        entity = WorkflowEntity(run_id="test-run-456")
        ctx = WorkflowContext(workflow_entity=entity, run_id="test-run-456")

        # Inject user response (simulating resume)
        entity.inject_user_response("yes")

        # Second call should return cached response without raising
        response = await ctx.wait_for_user(
            question="Approve this action?",
            input_type="approval",
            options=[
                {"id": "yes", "label": "Approve"},
                {"id": "no", "label": "Reject"},
            ],
        )

        assert response == "yes"

    @with_entity_context
    async def test_wait_for_user_text_input(self):
        """Test wait_for_user with text input type."""
        entity = WorkflowEntity(run_id="test-run-text")
        ctx = WorkflowContext(workflow_entity=entity, run_id="test-run-text")

        # Should raise exception with text input type
        with pytest.raises(WaitingForUserInputException) as exc_info:
            await ctx.wait_for_user(question="What is your name?", input_type="text")

        exc = exc_info.value
        assert exc.question == "What is your name?"
        assert exc.input_type == "text"
        assert exc.options == []

    @with_entity_context
    async def test_wait_for_user_choice_input(self):
        """Test wait_for_user with choice input type."""
        entity = WorkflowEntity(run_id="test-run-choice")
        ctx = WorkflowContext(workflow_entity=entity, run_id="test-run-choice")

        # Should raise exception with choice input type
        with pytest.raises(WaitingForUserInputException) as exc_info:
            await ctx.wait_for_user(
                question="Which model?",
                input_type="choice",
                options=[
                    {"id": "gpt4", "label": "GPT-4"},
                    {"id": "claude", "label": "Claude"},
                ],
            )

        exc = exc_info.value
        assert exc.question == "Which model?"
        assert exc.input_type == "choice"
        assert len(exc.options) == 2


class TestWorkflowEntityInjection:
    """Test user response injection in WorkflowEntity."""

    def test_inject_user_response_stores_correctly(self):
        """Test that inject_user_response stores response in completed steps."""
        entity = WorkflowEntity(run_id="test-inject-123")

        # Inject response
        entity.inject_user_response("approved")

        # Verify it's stored with correct key
        response_key = f"user_response:{entity.run_id}"
        assert entity.has_completed_step(response_key)
        assert entity.get_completed_step(response_key) == "approved"

    def test_inject_user_response_different_values(self):
        """Test injection with different response values."""
        entity = WorkflowEntity(run_id="test-inject-456")

        # Test various response values
        test_values = ["yes", "no", "San Francisco", "gpt4", "claude"]

        for value in test_values:
            entity = WorkflowEntity(run_id=f"test-{value}")
            entity.inject_user_response(value)

            response_key = f"user_response:{entity.run_id}"
            assert entity.get_completed_step(response_key) == value


class TestWorkflowWithWaitForUser:
    """Test complete workflows using wait_for_user."""

    @with_entity_context
    async def test_workflow_pauses_on_user_input(self):
        """Test that workflow pauses when wait_for_user is called."""

        @workflow(name="approval_workflow_pause_test")
        async def approval_workflow(ctx: WorkflowContext, data: str):
            # Process data
            processed = data.upper()

            # Ask for approval
            decision = await ctx.wait_for_user(
                f"Approve {processed}?",
                input_type="approval",
                options=[
                    {"id": "yes", "label": "Approve"},
                    {"id": "no", "label": "Reject"},
                ],
            )

            # Continue based on decision
            if decision == "yes":
                return f"Approved: {processed}"
            return "Rejected"

        # Create workflow context
        entity = WorkflowEntity(run_id="test-workflow-pause")
        ctx = WorkflowContext(workflow_entity=entity, run_id="test-workflow-pause")

        # First execution should pause
        with pytest.raises(WaitingForUserInputException) as exc_info:
            await approval_workflow(ctx, "hello")

        exc = exc_info.value
        assert "Approve HELLO?" in exc.question

    @with_entity_context
    async def test_workflow_resumes_with_user_response(self):
        """Test that workflow resumes correctly with user response."""

        @workflow(name="approval_workflow_resume_test")
        async def approval_workflow(ctx: WorkflowContext, data: str):
            # Process data
            processed = data.upper()

            # Ask for approval
            decision = await ctx.wait_for_user(
                f"Approve {processed}?",
                input_type="approval",
                options=[
                    {"id": "yes", "label": "Approve"},
                    {"id": "no", "label": "Reject"},
                ],
            )

            # Continue based on decision
            if decision == "yes":
                return f"Approved: {processed}"
            return "Rejected"

        # Create workflow context with injected response
        entity = WorkflowEntity(run_id="test-workflow-resume")
        entity.inject_user_response("yes")
        ctx = WorkflowContext(workflow_entity=entity, run_id="test-workflow-resume")

        # Should complete without raising
        result = await approval_workflow(ctx, "hello")
        assert result == "Approved: HELLO"

    @with_entity_context
    async def test_workflow_multiple_user_inputs(self):
        """Test workflow with multiple wait_for_user calls."""

        @workflow(name="multi_input_workflow_test")
        async def multi_input_workflow(ctx: WorkflowContext):
            # First input
            name = await ctx.wait_for_user("What is your name?", input_type="text")

            # Second input
            city = await ctx.wait_for_user("Which city?", input_type="text")

            return f"{name} from {city}"

        # Test first pause
        entity = WorkflowEntity(run_id="test-multi-1")
        ctx = WorkflowContext(workflow_entity=entity, run_id="test-multi-1")

        with pytest.raises(WaitingForUserInputException) as exc_info:
            await multi_input_workflow(ctx)

        assert exc_info.value.question == "What is your name?"

        # Note: Testing complete multi-pause flow requires integration tests
        # because each pause needs platform coordination


class TestExceptionAttributes:
    """Test WaitingForUserInputException attributes."""

    def test_exception_message(self):
        """Test exception message formatting."""
        exc = WaitingForUserInputException(
            question="Test question?",
            input_type="text",
            options=None,
            checkpoint_state={},
        )

        assert "Test question?" in str(exc)

    def test_exception_all_attributes(self):
        """Test all exception attributes are set correctly."""
        checkpoint = {"step1": "result1"}
        options = [{"id": "a", "label": "Option A"}]

        exc = WaitingForUserInputException(
            question="Choose one",
            input_type="choice",
            options=options,
            checkpoint_state=checkpoint,
        )

        assert exc.question == "Choose one"
        assert exc.input_type == "choice"
        assert exc.options == options
        assert exc.checkpoint_state == checkpoint

    def test_exception_with_no_options(self):
        """Test exception with None options defaults to empty list."""
        exc = WaitingForUserInputException(
            question="Text input",
            input_type="text",
            options=None,
            checkpoint_state={},
        )

        assert exc.options == []
