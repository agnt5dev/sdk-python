#!/usr/bin/env python3
"""
Memory Architecture Test Script

Tests entity memory functionality with session-scoped and user-scoped memory.

Usage:
    python test_memory.py session    # Test session memory
    python test_memory.py user       # Test user memory
    python test_memory.py multi      # Test multi-agent session
    python test_memory.py all        # Run all tests
"""

import sys
import json
from agnt5 import Client

GATEWAY_URL = "http://localhost:34181"


def test_session_memory():
    """Test session-scoped memory across multiple invocations."""
    print("\n" + "="*60)
    print("TEST 1: Session-Scoped Memory")
    print("="*60)

    client = Client(GATEWAY_URL)
    session_id = "test-session-123"

    # First interaction - establish conversation
    print(f"\n📝 Call 1: Introducing myself (session_id={session_id})")
    result1 = client.run(
        "test_session_memory_workflow",
        {"message": "Hello! My name is Alice and I'm a data scientist."},
        component_type="workflow",
        session_id=session_id,
    )
    print(f"Response: {result1['response']}")
    print(f"Message count: {result1['message_count']}")
    print(f"Memory scope: {result1['memory_scope']}")

    # Second interaction - test memory recall
    print(f"\n📝 Call 2: Testing recall (session_id={session_id})")
    result2 = client.run(
        "test_session_memory_workflow",
        {"message": "What's my name and profession?"},
        component_type="workflow",
        session_id=session_id,  # Same session!
    )
    print(f"Response: {result2['response']}")
    print(f"Message count: {result2['message_count']}")

    # Third interaction - add more context
    print(f"\n📝 Call 3: Adding more context (session_id={session_id})")
    result3 = client.run(
        "test_session_memory_workflow",
        {"message": "I'm working on a machine learning project about image classification."},
        component_type="workflow",
        session_id=session_id,
    )
    print(f"Response: {result3['response']}")
    print(f"Message count: {result3['message_count']}")

    # Fourth interaction - verify full conversation memory
    print(f"\n📝 Call 4: Recalling full conversation (session_id={session_id})")
    result4 = client.run(
        "test_session_memory_workflow",
        {"message": "Can you summarize what you know about me?"},
        component_type="workflow",
        session_id=session_id,
    )
    print(f"Response: {result4['response']}")
    print(f"Message count: {result4['message_count']}")

    # Verify with different session (should NOT remember)
    new_session = "test-session-456"
    print(f"\n📝 Call 5: Different session (session_id={new_session})")
    result5 = client.run(
        "test_session_memory_workflow",
        {"message": "What's my name?"},
        component_type="workflow",
        session_id=new_session,  # Different session!
    )
    print(f"Response: {result5['response']}")
    print(f"Message count: {result5['message_count']} (should be low)")

    print("\n✅ Session memory test complete!")
    return result4['message_count'] >= 7  # Should have accumulated messages


def test_user_memory():
    """Test user-scoped memory across different sessions."""
    print("\n" + "="*60)
    print("TEST 2: User-Scoped Memory")
    print("="*60)

    client = Client(GATEWAY_URL)
    user_id = "user-alice"

    # First session - set preference
    print(f"\n📝 Session 1: Setting preference (user_id={user_id})")
    result1 = client.run(
        "test_user_memory_workflow",
        {"message": "I prefer Python programming and use VS Code as my editor."},
        component_type="workflow",
        session_id="sess-001",
        user_id=user_id,
    )
    print(f"Response: {result1['response']}")
    print(f"Memory scope: {result1['memory_scope']}")

    # Second session - recall preference (DIFFERENT session, SAME user)
    print(f"\n📝 Session 2: Recalling preference (user_id={user_id}, different session)")
    result2 = client.run(
        "test_user_memory_workflow",
        {"message": "What programming language and editor do I prefer?"},
        component_type="workflow",
        session_id="sess-002",  # Different session!
        user_id=user_id,         # Same user!
    )
    print(f"Response: {result2['response']}")
    print(f"Message count: {result2['message_count']}")

    # Third session - add more preferences
    print(f"\n📝 Session 3: Adding more info (user_id={user_id})")
    result3 = client.run(
        "test_user_memory_workflow",
        {"message": "I also like working with pandas and numpy libraries."},
        component_type="workflow",
        session_id="sess-003",  # Yet another session!
        user_id=user_id,
    )
    print(f"Response: {result3['response']}")

    # Fourth session - verify accumulated user knowledge
    print(f"\n📝 Session 4: Full recall (user_id={user_id})")
    result4 = client.run(
        "test_user_memory_workflow",
        {"message": "What are all my programming preferences you know?"},
        component_type="workflow",
        session_id="sess-004",  # Different session again!
        user_id=user_id,
    )
    print(f"Response: {result4['response']}")
    print(f"Message count: {result4['message_count']}")

    # Different user (should NOT remember Alice's preferences)
    print(f"\n📝 Different user: user-bob")
    result5 = client.run(
        "test_user_memory_workflow",
        {"message": "What programming language do I prefer?"},
        component_type="workflow",
        session_id="sess-005",
        user_id="user-bob",  # Different user!
    )
    print(f"Response: {result5['response']}")
    print(f"Message count: {result5['message_count']} (should be low)")

    print("\n✅ User memory test complete!")
    return result4['message_count'] >= 7  # Should have accumulated across sessions


def test_multi_agent_session():
    """Test multiple agents sharing session memory."""
    print("\n" + "="*60)
    print("TEST 3: Multi-Agent Session Collaboration")
    print("="*60)

    client = Client(GATEWAY_URL)
    session_id = "collab-session-789"

    print(f"\n📝 Running multi-agent workflow (session_id={session_id})")
    result = client.run(
        "test_multi_agent_session_workflow",
        {"task": "Analyze the impact of AI on software development"},
        component_type="workflow",
        session_id=session_id,
    )

    print(f"\n🔬 Research Agent Output:")
    print(f"  {result['research'][:200]}...")

    print(f"\n📊 Analyst Agent Output:")
    print(f"  {result['analysis'][:200]}...")

    print(f"\n🤝 Agents in session: {result['agents_in_session']}")
    print(f"Memory scope: {result['memory_scope']}")

    print("\n✅ Multi-agent test complete!")
    return len(result['agents_in_session']) >= 2


def run_all_tests():
    """Run all memory tests."""
    print("\n" + "="*60)
    print("RUNNING ALL MEMORY ARCHITECTURE TESTS")
    print("="*60)

    results = {
        "session_memory": False,
        "user_memory": False,
        "multi_agent": False,
    }

    try:
        results["session_memory"] = test_session_memory()
    except Exception as e:
        print(f"\n❌ Session memory test failed: {e}")

    try:
        results["user_memory"] = test_user_memory()
    except Exception as e:
        print(f"\n❌ User memory test failed: {e}")

    try:
        results["multi_agent"] = test_multi_agent_session()
    except Exception as e:
        print(f"\n❌ Multi-agent test failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    test_type = sys.argv[1].lower()

    try:
        if test_type == "session":
            test_session_memory()
        elif test_type == "user":
            test_user_memory()
        elif test_type == "multi":
            test_multi_agent_session()
        elif test_type == "all":
            success = run_all_tests()
            sys.exit(0 if success else 1)
        else:
            print(f"Unknown test type: {test_type}")
            print(__doc__)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
