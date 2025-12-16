"""
Platform Verification Utilities

Helper functions for verifying platform state directly by querying
the database. These utilities enable tests to verify that state is
actually persisted to the platform, not just stored in-memory.

Supports multiple backends:
- SQLite (embedded mode)
- PostgreSQL (postgres mode)
- CockroachDB (managed mode - PostgreSQL compatible)

Functions:
- get_entity_state_from_platform: Query entity state from database
- get_workflow_run_status: Query workflow run status
- get_step_execution_count: Count step executions for idempotency checks
- wait_for_run_completion: Poll for run completion
"""

import json
import sqlite3
import time
from typing import Dict, Optional

import psycopg2


# ==================== Backend Detection ====================


def _detect_backend(db_url: str) -> str:
    """
    Detect database backend from URL.

    Returns:
        "sqlite" | "postgres" | "cockroach"
    """
    if db_url.endswith(".db") or "sqlite" in db_url.lower():
        return "sqlite"
    elif "cockroach" in db_url.lower():
        return "cockroach"
    elif "postgresql://" in db_url or "postgres://" in db_url:
        return "postgres"
    else:
        raise ValueError(f"Unknown database backend from URL: {db_url}")


def _get_connection(db_url: str):
    """
    Get database connection for given URL.

    Supports SQLite, PostgreSQL, and CockroachDB.
    """
    backend = _detect_backend(db_url)

    if backend == "sqlite":
        return sqlite3.connect(db_url)
    else:
        # PostgreSQL and CockroachDB use same driver
        return psycopg2.connect(db_url)


def get_entity_state_from_platform(
    db_url: str,
    entity_type: str,
    key: str
) -> Optional[Dict]:
    """
    Query database directly for entity state.

    Works with SQLite, PostgreSQL, and CockroachDB.

    This verifies state was actually persisted to platform,
    not just stored in-memory in the worker.

    Args:
        db_url: Database connection string (SQLite path or PostgreSQL URL)
        entity_type: Entity type (e.g., "ShoppingCart")
        key: Entity key (e.g., "user-123")

    Returns:
        Entity state dict if found, None otherwise

    Example:
        # SQLite
        state = get_entity_state_from_platform(
            db_url="/data/orchestration.db",
            entity_type="ShoppingCart",
            key="user-123"
        )

        # PostgreSQL/CockroachDB
        state = get_entity_state_from_platform(
            db_url="postgresql://root@localhost:26257/orchestration",
            entity_type="ShoppingCart",
            key="user-123"
        )
    """
    conn = _get_connection(db_url)
    cursor = conn.cursor()

    try:
        backend = _detect_backend(db_url)

        # Query entities table (schema from migrations)
        if backend == "sqlite":
            # SQLite uses ? placeholders
            cursor.execute(
                """
                SELECT current_state, state_version, updated_at
                FROM entities
                WHERE entity_type = ? AND entity_key = ?
                LIMIT 1
                """,
                (entity_type, key)
            )
        else:
            # PostgreSQL/CockroachDB use %s placeholders
            cursor.execute(
                """
                SELECT current_state, state_version, updated_at
                FROM entities
                WHERE entity_type = %s AND entity_key = %s
                LIMIT 1
                """,
                (entity_type, key)
            )

        row = cursor.fetchone()

        if row:
            current_state, state_version, updated_at = row
            return {
                "state": json.loads(current_state) if isinstance(current_state, str) else current_state,
                "version": state_version,
                "updated_at": updated_at
            }

        return None

    finally:
        cursor.close()
        conn.close()


def get_workflow_run_status(db_url: str, run_id: str) -> Optional[Dict]:
    """
    Get workflow run status from platform database.

    Works with SQLite, PostgreSQL, and CockroachDB.

    Args:
        db_url: Database connection string
        run_id: Run ID returned from client.submit()

    Returns:
        Run status dict if found, None otherwise

    Example:
        status = get_workflow_run_status(
            db_url="postgresql://root@localhost:26257/orchestration",
            run_id="run-abc-123"
        )
        assert status["status"] == "completed"
    """
    conn = _get_connection(db_url)
    cursor = conn.cursor()

    try:
        backend = _detect_backend(db_url)

        # Query runs table
        # Note: Actual schema may differ - adjust based on platform implementation
        if backend == "sqlite":
            cursor.execute(
                """
                SELECT
                    status,
                    component_name,
                    component_type,
                    created_at,
                    updated_at,
                    completed_at
                FROM runs
                WHERE id = ?
                """,
                (run_id,)
            )
        else:
            cursor.execute(
                """
                SELECT
                    status,
                    component_name,
                    component_type,
                    created_at,
                    updated_at,
                    completed_at
                FROM runs
                WHERE id = %s
                """,
                (run_id,)
            )

        row = cursor.fetchone()

        if row:
            (status, component_name, component_type,
             created_at, updated_at, completed_at) = row

            return {
                "status": status,
                "component_name": component_name,
                "component_type": component_type,
                "created_at": created_at,
                "updated_at": updated_at,
                "completed_at": completed_at,
            }

        return None

    finally:
        cursor.close()
        conn.close()


def get_step_execution_count(
    db_url: str,
    run_id: str,
    step_name: str
) -> int:
    """
    Count how many times a workflow step was executed.

    Works with SQLite, PostgreSQL, and CockroachDB.

    Used to verify idempotency - steps should only execute once
    even if workflow is replayed after crash.

    Args:
        db_url: Database connection string
        run_id: Run ID
        step_name: Step name to count

    Returns:
        Number of times step was executed

    Example:
        count = get_step_execution_count(
            db_url="postgresql://root@localhost:26257/orchestration",
            run_id="run-abc-123",
            step_name="step_0"
        )
        assert count == 1, "Step should execute exactly once"
    """
    conn = _get_connection(db_url)
    cursor = conn.cursor()

    try:
        backend = _detect_backend(db_url)

        # Query workflow steps/checkpoints table
        # Note: Actual schema may differ - adjust based on platform implementation
        if backend == "sqlite":
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM workflow_steps
                WHERE run_id = ? AND step_name = ?
                """,
                (run_id, step_name)
            )
        else:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM workflow_steps
                WHERE run_id = %s AND step_name = %s
                """,
                (run_id, step_name)
            )

        count = cursor.fetchone()[0]
        return count

    finally:
        cursor.close()
        conn.close()


def wait_for_run_completion(
    db_url: str,
    run_id: str,
    timeout: int = 30,
    poll_interval: float = 0.5
) -> Optional[Dict]:
    """
    Poll database until run completes or timeout.

    Works with SQLite, PostgreSQL, and CockroachDB.

    Args:
        db_url: Database connection string
        run_id: Run ID to wait for
        timeout: Maximum seconds to wait
        poll_interval: Seconds between polls

    Returns:
        Final run status dict, or None if timeout

    Example:
        status = wait_for_run_completion(
            db_url="postgresql://root@localhost:26257/orchestration",
            run_id="run-abc-123",
            timeout=30
        )
        assert status["status"] == "completed"
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = get_workflow_run_status(db_url, run_id)

        if status and status["status"] in ["completed", "failed", "cancelled"]:
            return status

        time.sleep(poll_interval)

    return None


def verify_entity_state_exists(
    db_url: str,
    entity_type: str,
    key: str
) -> bool:
    """
    Check if entity state exists in platform database.

    Works with SQLite, PostgreSQL, and CockroachDB.

    Args:
        db_url: Database connection string
        entity_type: Entity type
        key: Entity key

    Returns:
        True if state exists, False otherwise
    """
    state = get_entity_state_from_platform(db_url, entity_type, key)
    return state is not None


def get_run_count_by_status(db_url: str, status: str) -> int:
    """
    Count runs by status.

    Works with SQLite, PostgreSQL, and CockroachDB.

    Args:
        db_url: Database connection string
        status: Status to count (e.g., "completed", "failed", "running")

    Returns:
        Number of runs with given status
    """
    conn = _get_connection(db_url)
    cursor = conn.cursor()

    try:
        backend = _detect_backend(db_url)

        if backend == "sqlite":
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM runs
                WHERE status = ?
                """,
                (status,)
            )
        else:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM runs
                WHERE status = %s
                """,
                (status,)
            )

        count = cursor.fetchone()[0]
        return count

    finally:
        cursor.close()
        conn.close()


def clear_test_data(db_url: str, tenant_id: str = "test-tenant-001"):
    """
    Clear test data from platform database.

    Works with SQLite, PostgreSQL, and CockroachDB.

    WARNING: Only use in test cleanup! This deletes data.

    Args:
        db_url: Database connection string
        tenant_id: Tenant ID to clear (default: test tenant)
    """
    conn = _get_connection(db_url)
    cursor = conn.cursor()

    try:
        backend = _detect_backend(db_url)

        if backend == "sqlite":
            # Clear entity states
            cursor.execute(
                "DELETE FROM entity_states WHERE tenant_id = ?",
                (tenant_id,)
            )

            # Clear runs
            cursor.execute(
                "DELETE FROM runs WHERE tenant_id = ?",
                (tenant_id,)
            )

            # Clear workflow steps
            cursor.execute(
                "DELETE FROM workflow_steps WHERE tenant_id = ?",
                (tenant_id,)
            )
        else:
            # Clear entity states
            cursor.execute(
                "DELETE FROM entity_states WHERE tenant_id = %s",
                (tenant_id,)
            )

            # Clear runs
            cursor.execute(
                "DELETE FROM runs WHERE tenant_id = %s",
                (tenant_id,)
            )

            # Clear workflow steps
            cursor.execute(
                "DELETE FROM workflow_steps WHERE tenant_id = %s",
                (tenant_id,)
            )

        conn.commit()

    finally:
        cursor.close()
        conn.close()


def get_db_path_for_platform(platform: Dict) -> str:
    """
    Get the correct database path for the current platform mode.

    Handles the different database path patterns across modes:
    - Local: /tmp/agnt5/<service-name>/agnt5.db
    - Subprocess: {data_dir}/orchestration.db
    - Others: Use db_url directly

    Args:
        platform: Platform configuration dict from fixture

    Returns:
        Database path suitable for direct SQLite/PostgreSQL connection
    """
    mode = platform.get("mode")

    if mode == "local":
        # Local mode: DB is at /tmp/agnt5/<service>/agnt5.db
        # The service name is "agnt5-python-examples" when running with `just platform standalone python`
        service_name = "agnt5-python-examples"
        return f"/tmp/agnt5/{service_name}/agnt5.db"
    elif mode == "subprocess":
        # Subprocess mode: use host_db_path or db_url
        return platform.get("host_db_path") or platform.get("db_url")
    else:
        # Other modes (embedded, postgres, managed): use db_url
        return platform.get("db_url")
