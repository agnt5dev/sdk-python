#!/usr/bin/env python3
"""
Verify Crash Recovery for AGNT5

Simple script to check if a run completed after crash recovery.

Usage:
    python verify_recovery.py <run_id>
"""

import sqlite3
import sys
from pathlib import Path


def query_db(query: str) -> list:
    """Query SQLite database."""
    db_path = "/tmp/agnt5-dev-data/agnt5-dev-orchestration.db"

    if not Path(db_path).exists():
        print(f"❌ Database not found at {db_path}")
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"❌ Database query error: {e}")
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_recovery.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]

    print(f"🔍 Checking recovery status for run: {run_id}")
    print("=" * 60)

    # Check run status
    runs = query_db(
        f"""
        SELECT id, component_name, status, error_message,
               submitted_at, started_at, completed_at
        FROM runs
        WHERE id = '{run_id}'
    """
    )

    if not runs:
        print(f"❌ Run {run_id} not found in database")
        sys.exit(1)

    run = runs[0]
    run_id, component, status, error_msg, submitted, started, completed = run

    print(f"\n📊 Run Details:")
    print(f"   ID: {run_id}")
    print(f"   Component: {component}")
    print(f"   Status: {status}")
    print(f"   Submitted: {submitted}")
    print(f"   Started: {started}")
    print(f"   Completed: {completed}")

    if error_msg:
        print(f"   Error: {error_msg}")

    # Check for output
    outputs = query_db(
        f"""
        SELECT output_data, output_type, created_at
        FROM run_outputs
        WHERE run_id = '{run_id}'
    """
    )

    if outputs:
        print(f"\n📤 Output:")
        for output_data, output_type, created_at in outputs:
            print(f"   Type: {output_type}")
            print(f"   Data: {output_data}")
            print(f"   Created: {created_at}")
    else:
        print(f"\n⚠️  No output found (run may not be complete)")

    # Check events for this run
    events = query_db(
        f"""
        SELECT event_type, created_at
        FROM journal_events
        WHERE aggregate_id = '{run_id}'
        ORDER BY sequence_num
    """
    )

    if events:
        print(f"\n📝 Event Timeline:")
        for event_type, created_at in events:
            print(f"   {created_at} - {event_type}")

    # Final verdict
    print("\n" + "=" * 60)
    if status == "completed" and outputs:
        print("✅ RECOVERY SUCCESSFUL!")
        print("   Run completed and output was persisted after crash.")
    elif status == "completed":
        print("⚠️  PARTIAL SUCCESS")
        print("   Run completed but output not found.")
    elif status == "failed":
        print("❌ RUN FAILED")
        print(f"   Error: {error_msg}")
    elif status == "running":
        print("⏳ STILL RUNNING")
        print("   Run is still in progress. Wait or check worker logs.")
    else:
        print(f"⚠️  UNKNOWN STATUS: {status}")


if __name__ == "__main__":
    main()
