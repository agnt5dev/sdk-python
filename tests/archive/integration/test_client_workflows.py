"""
Integration Tests: Workflow Recovery

Tests workflow execution, checkpointing, and crash recovery using the Client API.

These tests validate:
- Basic workflow execution works end-to-end
- Workflows resume after worker crashes
- Steps are not re-executed (idempotency)
- Workflow state persists to platform

Expected Results (Week 1 - RED Phase):
- test_workflow_basic_execution: ✅ PASS (basic execution works)
- test_workflow_resumes_after_crash: ❌ FAIL (TODO: checkpoint persistence)
- test_workflow_idempotency: ❌ FAIL (TODO: step replay logic)
"""

import time

import pytest

from conftest import restart_worker


@pytest.mark.integration
def test_workflow_basic_execution(client, worker_process):
    """
    Test basic workflow execution via client.

    This should PASS - validates basic workflow orchestration.
    """
    result = client.run("order_fulfillment", {"order_id": "order-123"}, component_type="workflow")

    assert result["status"] == "completed"
    assert result["order_id"] == "order-123"
    assert "payment_id" in result
    assert result["payment_id"] == "pay_order-123"
    assert "tracking_number" in result
    assert result["tracking_number"] == "TRACK_order-123"


@pytest.mark.integration
def test_workflow_with_state_management(client, worker_process):
    """
    Test workflow state management across steps.

    This should PASS - validates ctx.state API works.
    """
    result = client.run("order_fulfillment", {"order_id": "order-456"}, component_type="workflow")

    # Workflow should complete successfully
    assert result["status"] == "completed"

    # TODO: Query platform to verify intermediate state was saved
    # For now, just verify final result is correct


@pytest.mark.integration
@pytest.mark.slow
def test_workflow_resumes_after_crash(client, worker_process, platform):
    """
    Test workflow checkpoint replay after worker crash.

    ❌ Expected to FAIL - exposes missing checkpoint persistence.

    Failure indicates:
    - Workflow checkpoints are not saved to platform
    - Worker crash causes workflow to restart from beginning
    - Steps may be re-executed (breaking idempotency)

    This test drives implementation of workflow checkpointing in Week 2.
    """
    # 1. Submit long workflow
    run_id = client.submit("long_workflow", {"steps": 5}, component_type="workflow")

    # 2. Wait for partial completion (step 2)
    time.sleep(2.5)

    # 3. Crash worker
    worker_process.kill()
    worker_process.wait()

    # 4. Start new worker
    new_worker = restart_worker(worker_process, platform)

    # 5. Workflow should resume and complete
    # ❌ This will likely FAIL or timeout
    # Without checkpoint persistence, workflow can't resume
    result = client.wait_for_result(run_id, timeout=30)

    assert result["status"] == "completed", (
        f"Expected workflow to resume after crash, but got status: {result.get('status')}. "
        "Workflow checkpoints are not being persisted to platform. "
        "Workers cannot resume workflows after crashes."
    )

    # 6. Verify all steps completed
    assert result["steps_completed"] == 5

    # Cleanup
    new_worker.terminate()


@pytest.mark.integration
@pytest.mark.skip(reason="Step execution tracking not yet implemented")
def test_workflow_idempotency(client, worker_process, platform):
    """
    Test workflow steps are not re-executed during replay.

    ⏭️  SKIPPED - Requires platform verification utilities.

    This validates that:
    - Steps that completed before crash are not re-executed
    - Only remaining steps are executed
    - Idempotency is maintained
    """
    from tests.integration.utils import get_step_execution_count

    # 1. Submit workflow
    run_id = client.submit("long_workflow", {"steps": 5}, component_type="workflow")

    # 2. Wait for partial completion
    time.sleep(2.5)

    # 3. Crash and restart worker
    worker_process.kill()
    worker_process.wait()
    new_worker = restart_worker(worker_process, platform)

    # 4. Wait for completion
    result = client.wait_for_result(run_id, timeout=30)
    assert result["status"] == "completed"

    # 5. Verify each step was executed exactly once
    for step_num in range(5):
        execution_count = get_step_execution_count(
            db_url=platform["db_url"],
            run_id=run_id,
            step_name=f"step_{step_num}"
        )

        assert execution_count == 1, (
            f"Step {step_num} was executed {execution_count} times, expected 1. "
            "Steps should not be re-executed during replay."
        )

    # Cleanup
    new_worker.terminate()


@pytest.mark.integration
def test_workflow_async_submit_and_wait(client, worker_process):
    """
    Test async workflow submission and polling.

    This should PASS - validates client.submit() and client.wait_for_result().
    """
    # Submit workflow asynchronously
    run_id = client.submit("order_fulfillment", {"order_id": "order-789"}, component_type="workflow")

    assert run_id is not None
    assert isinstance(run_id, str)

    # Wait for completion
    result = client.wait_for_result(run_id, timeout=10)

    assert result["status"] == "completed"
    assert result["order_id"] == "order-789"


@pytest.mark.integration
def test_workflow_parallel_execution(client, worker_process):
    """
    Test multiple workflows can execute in parallel.

    This should PASS - validates concurrent workflow execution.
    """
    # Submit multiple workflows
    run_ids = []
    for i in range(5):
        run_id = client.submit("order_fulfillment", {"order_id": f"order-{i}"}, component_type="workflow")
        run_ids.append(run_id)

    # Wait for all to complete
    results = []
    for run_id in run_ids:
        result = client.wait_for_result(run_id, timeout=10)
        results.append(result)

    # Verify all completed
    assert len(results) == 5
    for i, result in enumerate(results):
        assert result["status"] == "completed"
        assert result["order_id"] == f"order-{i}"


@pytest.mark.integration
@pytest.mark.skip(reason="Workflow state query API not yet implemented")
def test_workflow_state_query(client, worker_process, platform):
    """
    Test querying workflow state mid-execution.

    ⏭️  SKIPPED - Requires workflow state query API.

    This validates:
    - Can query workflow state while running
    - State shows current step and progress
    - State is accurate and up-to-date
    """
    from tests.integration.utils import get_workflow_run_status

    # Submit long workflow
    run_id = client.submit("long_workflow", {"steps": 5}, component_type="workflow")

    # Query state mid-execution
    time.sleep(2)

    status = get_workflow_run_status(
        db_url=platform["db_url"],
        run_id=run_id
    )

    assert status is not None
    assert status["status"] in ["running", "pending"]
    assert status["completed_steps"] < status["total_steps"]

    # Wait for completion
    client.wait_for_result(run_id, timeout=30)

    # Query final state
    final_status = get_workflow_run_status(
        db_url=platform["db_url"],
        run_id=run_id
    )

    assert final_status["status"] == "completed"
    assert final_status["completed_steps"] == final_status["total_steps"]
