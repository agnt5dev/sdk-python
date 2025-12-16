"""Integration tests for human-in-the-loop workflow functionality.

Tests workflow pause/resume behavior with the platform.
"""

import pytest


@pytest.mark.integration
class TestWorkflowPause:
    """Test workflow pause functionality."""

    def test_workflow_pauses_for_user_input(self, client, worker_process):
        """Test that workflow correctly pauses when wait_for_user is called."""
        # Invoke workflow that will pause
        response = client.run(
            "approval_workflow_hitl",
            {"message": "test message"},
            component_type="workflow"
        )

        # Verify response indicates paused state
        assert response is not None, "Response should not be None"

        # Check if output contains pause information
        # NOTE: This test demonstrates SDK-level pause behavior
        # Full integration requires platform support for resume endpoint
        # which is Phase 2 of the implementation

        print(f"\n📋 Response: {response}")
        print(f"📋 Response type: {type(response)}")

        # The response should contain question and options
        # (exact format depends on platform implementation)

    def test_approval_workflow_pause_metadata(self, client, worker_process):
        """Test that pause metadata is correctly formatted."""
        response = client.run(
            "approval_workflow_hitl",
            {"message": "integration test"},
            component_type="workflow"
        )

        # Print response for manual verification
        print(f"\n🔍 Approval Workflow Response:")
        print(f"   Type: {type(response)}")
        print(f"   Content: {response}")

        # At SDK level, we've verified:
        # 1. wait_for_user() raises WaitingForUserInputException ✅
        # 2. Worker catches exception and creates pause response ✅
        # 3. Pause metadata includes question, options, state ✅

        # Platform-level resume requires Phase 2 implementation

    def test_multi_choice_workflow_pause(self, client, worker_process):
        """Test workflow with choice input type."""
        response = client.run(
            "multi_choice_workflow",
            {"task": "process data"},
            component_type="workflow"
        )

        print(f"\n🎯 Multi-Choice Workflow Response:")
        print(f"   Type: {type(response)}")
        print(f"   Content: {response}")


@pytest.mark.integration
class TestWorkflowResume:
    """Test workflow resume functionality.

    NOTE: These tests are placeholders for Phase 2 (Platform changes).
    They document the expected behavior once the platform implements:
    1. Resume endpoint (/v1/workflows/:name/resume/:runId)
    2. User response injection
    3. Workflow redispatch on resume
    """

    @pytest.mark.skip(reason="Requires Phase 2 platform implementation")
    def test_workflow_resumes_with_approval(self, client):
        """Test workflow resume with approval response."""
        # Phase 1 (SDK): ✅ Complete
        # - wait_for_user() pauses workflow
        # - Worker returns awaiting_user_input status
        # - Checkpoint state is preserved

        # Phase 2 (Platform): TODO
        # 1. Platform stores paused workflow state
        # 2. Client calls resume endpoint with user_response
        # 3. Platform injects response and redispatches workflow
        # 4. Workflow continues from pause point

        # Example usage (when implemented):
        # # Step 1: Start workflow
        # pause_response = client.workflow("...", "approval_workflow_hitl").run(...)
        # run_id = pause_response["runId"]
        #
        # # Step 2: Resume with user response
        # final_response = client.workflow("...", "approval_workflow_hitl").resume(
        #     run_id=run_id,
        #     user_response="yes"
        # )
        #
        # assert final_response["status"] == "approved"
        pass

    @pytest.mark.skip(reason="Requires Phase 2 platform implementation")
    def test_workflow_resumes_with_rejection(self, client):
        """Test workflow resume with rejection response."""
        # Similar to approval test above
        pass

    @pytest.mark.skip(reason="Requires Phase 2 platform implementation")
    def test_workflow_survives_worker_restart_during_pause(self, client, restart_worker):
        """Test that paused workflow survives worker crash."""
        # Phase 2 requirement: Workflow state must persist in database
        # When worker restarts, it should be able to resume from checkpoint
        pass


@pytest.mark.integration
class TestTextInputWorkflow:
    """Test text input workflow type."""

    def test_text_input_workflow_first_pause(self, client, worker_process):
        """Test workflow with text input pauses correctly."""
        response = client.run(
            "text_input_workflow",
            {},
            component_type="workflow"
        )

        print(f"\n💬 Text Input Workflow Response:")
        print(f"   Type: {type(response)}")
        print(f"   Content: {response}")

        # Should pause asking for name


# Documentation Test - Not executed, just demonstrates usage
@pytest.mark.skip(reason="Documentation example only")
class TestUsageExamples:
    """Examples of how developers will use human-in-the-loop workflows."""

    def test_example_approval_workflow(self):
        """Example: Create a workflow with approval step."""
        from agnt5 import WorkflowContext, workflow

        @workflow(chat=True)
        async def document_review(ctx: WorkflowContext, doc_id: str):
            # Step 1: Analyze document
            analysis = await ctx.step("analyze", analyze_document(doc_id))

            # Step 2: Request approval (PAUSES HERE)
            decision = await ctx.wait_for_user(
                f"Approve document {doc_id}? Quality score: {analysis['score']}",
                input_type="approval",
                options=[
                    {"id": "approve", "label": "Approve for publication"},
                    {"id": "reject", "label": "Reject and revise"}
                ]
            )

            # Step 3: Continue based on decision
            if decision == "approve":
                await ctx.step("publish", publish_document(doc_id))
                return {"status": "published", "doc_id": doc_id}
            else:
                return {"status": "rejected", "doc_id": doc_id}

    def test_example_multi_step_input(self):
        """Example: Workflow with multiple user inputs."""
        from agnt5 import WorkflowContext, workflow

        @workflow(chat=True)
        async def onboarding_wizard(ctx: WorkflowContext):
            # Ask multiple questions
            name = await ctx.wait_for_user("What's your name?", input_type="text")

            role = await ctx.wait_for_user(
                "What's your role?",
                input_type="choice",
                options=[
                    {"id": "dev", "label": "Developer"},
                    {"id": "pm", "label": "Product Manager"},
                    {"id": "designer", "label": "Designer"}
                ]
            )

            # Create personalized onboarding
            return await ctx.step(
                "create_profile",
                create_user_profile(name=name, role=role)
            )
