#!/bin/bash
# Test script for memory scope verification
# Usage: ./test_memory_scopes.sh

set -e

BASE_URL="http://localhost:34181/v1/run/workflow"

echo "=== Testing Memory Scopes ==="
echo

# Test 1: Run-scoped (ephemeral)
echo "1️⃣  Testing RUN-SCOPED workflow (no session/user)..."
RUN1=$(curl -s -X POST "$BASE_URL/test_workflow" \
  -H "Content-Type: application/json" \
  -d '{}' | jq -r '.run_id')
echo "   ✓ Created run: $RUN1"
echo "   Expected entity_key: run:$RUN1"
echo

# Test 2: Session-scoped (first invocation)
echo "2️⃣  Testing SESSION-SCOPED workflow (session: my-session-123)..."
RUN2=$(curl -s -X POST "$BASE_URL/test_workflow" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session-123" \
  -d '{}' | jq -r '.run_id')
echo "   ✓ Created run: $RUN2"
echo "   Expected entity_key: session:my-session-123"
echo

# Test 3: Session-scoped (second invocation, same session)
echo "3️⃣  Testing SESSION-SCOPED workflow (SAME session: my-session-123)..."
RUN3=$(curl -s -X POST "$BASE_URL/test_workflow" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-session-123" \
  -d '{}' | jq -r '.run_id')
echo "   ✓ Created run: $RUN3"
echo "   Expected entity_key: session:my-session-123 (SAME as test 2)"
echo

# Test 4: User-scoped
echo "4️⃣  Testing USER-SCOPED workflow (user: user-alice)..."
RUN4=$(curl -s -X POST "$BASE_URL/test_workflow" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-alice" \
  -d '{}' | jq -r '.run_id')
echo "   ✓ Created run: $RUN4"
echo "   Expected entity_key: user:user-alice"
echo

# Test 5: User-scoped with different session (should use user key)
echo "5️⃣  Testing USER-SCOPED workflow (SAME user, DIFFERENT session)..."
RUN5=$(curl -s -X POST "$BASE_URL/test_workflow" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user-alice" \
  -H "X-Session-ID: another-session" \
  -d '{}' | jq -r '.run_id')
echo "   ✓ Created run: $RUN5"
echo "   Expected entity_key: user:user-alice (SAME as test 4, user takes priority)"
echo

echo "=== Verification ==="
echo "Query the database to verify entity keys:"
echo
echo "sqlite3 /tmp/agnt5/agnt5-python-test-bench/orchestration.db \\"
echo "  \"SELECT entity_key, entity_type, created_at FROM entities WHERE entity_type = 'WorkflowEntity' ORDER BY created_at DESC LIMIT 5;\""
echo
echo "Expected results:"
echo "  - 1 entity with key: run:$RUN1"
echo "  - 1 entity with key: session:my-session-123 (shared by runs $RUN2 and $RUN3)"
echo "  - 1 entity with key: user:user-alice (shared by runs $RUN4 and $RUN5)"
echo
echo "Total unique entities: 3 (not 5, because session and user are shared!)"
