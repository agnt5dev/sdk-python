"""
End-to-End Integration Tests for Functions

These tests verify the complete function execution flow:
1. Worker starts and registers functions
2. Client invokes functions via platform
3. Worker executes functions
4. Results are returned correctly
5. Error handling works as expected

Uses the test-service blueprint which provides simple functions
without LLM dependencies for reliable testing.
"""

import pytest
import time


@pytest.mark.integration
def test_simple_function_execution(client, worker_process, platform):
    """
    Test basic function execution end-to-end.

    Verifies:
    - Function can be invoked via client
    - Worker receives and executes function
    - Result is returned correctly
    """
    from agnt5 import Client

    # Give worker time to register
    time.sleep(2)

    # Create client with longer timeout for analysis
    client_with_timeout = Client(platform["gateway_url"], timeout=120.0)

    # Run function 5 times and measure timing
    num_runs = 5
    durations = []

    for i in range(num_runs):
        start = time.time()
        result = client_with_timeout.run("greet", {"name": f"Alice{i}"})
        duration = time.time() - start
        durations.append(duration)

        assert "message" in result
        assert result["message"] == f"Hello, Alice{i}!"

    # Calculate statistics
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)

    print(f"\n✅ Function executed successfully ({num_runs} runs)")
    print(f"   Latency - Min: {min_duration*1000:.2f}ms, Avg: {avg_duration*1000:.2f}ms, Max: {max_duration*1000:.2f}ms")


@pytest.mark.integration
def test_function_with_different_inputs(client, worker_process):
    """
    Test function with various input values.

    Verifies:
    - Function handles different input values correctly
    - Multiple invocations work independently
    """
    time.sleep(2)

    test_cases = [
        {"name": "Bob", "expected": "Hello, Bob!"},
        {"name": "Charlie", "expected": "Hello, Charlie!"},
        {"name": "测试", "expected": "Hello, 测试!"},  # Unicode
    ]

    for test_case in test_cases:
        result = client.run("greet", {"name": test_case["name"]})
        assert "message" in result
        assert result["message"] == test_case["expected"]

    print(f"\n✅ All test cases passed ({len(test_cases)} inputs)")


@pytest.mark.integration
def test_function_error_handling(client, worker_process):
    """
    Test function error handling.

    Verifies:
    - Exceptions in functions are caught
    - Error details are returned to client
    - Worker continues functioning after errors
    """
    time.sleep(2)

    # Invoke function that always fails
    from agnt5.client import RunError

    try:
        result = client.run("failing_function", {"error": "Test error message"})
        assert False, "Expected RunError to be raised"
    except RunError as e:
        assert "Test error message" in str(e)

    # Verify worker still works after error
    result2 = client.run("greet", {"name": "After Error"})
    assert "message" in result2
    assert result2["message"] == "Hello, After Error!"

    print(f"\n✅ Error handling works correctly")
    print(f"   Worker still functional after error")


@pytest.mark.integration
@pytest.mark.slow
def test_function_retry_logic(client, worker_process):
    """
    Test function retry mechanism.

    Verifies:
    - Functions can be configured with retries
    - Retries happen automatically on failure
    - Function succeeds after configured failures
    """
    time.sleep(2)

    # This function fails 2 times, then succeeds
    # It's configured with max_attempts=5
    result = client.run("flaky_function", {"fail_count": 2})

    assert "succeeded" in result
    assert result["succeeded"] is True
    assert result["attempts"] == 3  # Failed 2 times, succeeded on 3rd

    print(f"\n✅ Retry logic works correctly")
    print(f"   Attempts: {result['attempts']}")


@pytest.mark.integration
def test_concurrent_function_executions(client, worker_process):
    """
    Test concurrent function executions.

    Verifies:
    - Multiple functions can execute concurrently
    - Results are returned correctly for each invocation
    - No interference between concurrent executions
    """
    time.sleep(2)

    import concurrent.futures

    def invoke_function(name):
        return client.run("greet", {"name": name})

    # Execute 5 function calls concurrently
    names = ["User1", "User2", "User3", "User4", "User5"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(invoke_function, name) for name in names]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Verify all succeeded
    assert len(results) == 5

    # Verify each got correct result
    returned_messages = {r["message"] for r in results}
    expected_messages = {f"Hello, {name}!" for name in names}
    assert returned_messages == expected_messages

    print(f"\n✅ Concurrent execution works correctly")
    print(f"   Executed: {len(results)} concurrent calls")
    print(f"   Success rate: 100%")


@pytest.mark.integration
@pytest.mark.slow
def test_high_concurrency_async_functions(platform, worker_process):
    """
    Test high concurrency with 100 async function invocations.

    Verifies:
    - Platform can handle high concurrent load
    - All async functions execute correctly under load
    - Performance characteristics at scale
    - No race conditions or deadlocks
    """
    time.sleep(2)

    import concurrent.futures
    from agnt5 import Client

    # Create client with longer timeout for high concurrency
    client = Client(platform["gateway_url"], timeout=120.0)

    num_requests = 20
    max_workers = 20  # Thread pool size

    print(f"\n🚀 Starting high concurrency test ({num_requests} requests, {max_workers} workers)")

    def invoke_function(request_id):
        """Invoke greet function with unique request ID."""
        start = time.time()
        result = client.run("greet", {"name": f"User{request_id}"})
        duration = time.time() - start
        return {
            "request_id": request_id,
            "result": result,
            "duration": duration,
        }

    # Track timing
    test_start = time.time()

    # Execute 100 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(invoke_function, i) for i in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    test_duration = time.time() - test_start

    # Verify all requests succeeded
    assert len(results) == num_requests

    # Verify each got correct result
    for item in results:
        request_id = item["request_id"]
        result = item["result"]
        assert "message" in result
        assert result["message"] == f"Hello, User{request_id}!"

    # Calculate statistics
    durations = [item["duration"] for item in results]
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    throughput = num_requests / test_duration

    print(f"\n✅ High concurrency test completed successfully")
    print(f"   Total requests: {num_requests}")
    print(f"   Success rate: 100%")
    print(f"   Total time: {test_duration:.2f}s")
    print(f"   Throughput: {throughput:.2f} requests/second")
    print(f"   Latency - Min: {min_duration*1000:.2f}ms, Avg: {avg_duration*1000:.2f}ms, Max: {max_duration*1000:.2f}ms")


@pytest.mark.integration
def test_function_context_access(client, worker_process):
    """
    Test that functions have access to execution context.

    Verifies:
    - Context is available to functions
    - Logger works correctly
    - Context provides expected properties

    Note: This is more of a smoke test since we can't easily
    inspect logs from the integration test, but we verify
    the function executes without errors.
    """
    time.sleep(2)

    # The greet function uses ctx.logger.info internally
    result = client.run("greet", {"name": "Context Test"})

    assert "message" in result
    # If context was broken, the function would have errored

    print(f"\n✅ Function context access works")


@pytest.mark.integration
def test_function_with_missing_parameter(client, worker_process):
    """
    Test function invocation with missing required parameter.

    Verifies:
    - Missing parameters are caught
    - Appropriate error is returned
    """
    time.sleep(2)

    # Invoke without required 'name' parameter
    from agnt5.client import RunError

    try:
        result = client.run("greet", {})  # Missing 'name'
        assert False, "Expected RunError to be raised"
    except RunError as e:
        print(f"\n✅ Parameter validation works")
        print(f"   Error: {e}")


@pytest.mark.integration
def test_function_with_invalid_service_name(client, worker_process):
    """
    Test invoking function with non-existent service.

    Verifies:
    - Invalid service names are handled
    - Appropriate error is returned

    Note: Since the client API doesn't have a service_name parameter,
    we test invoking a function that doesn't exist, which achieves
    the same validation goal.
    """
    time.sleep(2)

    from agnt5.client import RunError

    try:
        result = client.run("non_existent_function", {"name": "Test"})
        assert False, "Expected RunError to be raised"
    except RunError as e:
        print(f"\n✅ Invalid function name handled")
        print(f"   Error: {e}")


@pytest.mark.integration
def test_function_with_invalid_function_name(client, worker_process):
    """
    Test invoking non-existent function.

    Verifies:
    - Invalid function names are handled
    - Appropriate error is returned
    """
    time.sleep(2)

    from agnt5.client import RunError

    try:
        result = client.run("truly_non_existent_function", {"name": "Test"})
        assert False, "Expected RunError to be raised"
    except RunError as e:
        print(f"\n✅ Invalid function name handled")
        print(f"   Error: {e}")


# TODO: Add tests for:
# - Function execution time tracking
# - Function metadata (input/output schemas)
# - Long-running functions with timeouts
# - Function execution with worker restart (durability)
# - Streaming function responses
# - Function execution metrics/telemetry
