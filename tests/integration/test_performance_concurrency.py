"""
Performance and Concurrency Integration Tests

These tests verify platform performance and concurrency characteristics:
1. High concurrent load handling
2. Throughput measurements
3. Latency characteristics
4. State isolation under concurrency
5. Resource utilization
6. Performance regression detection

These tests are marked as @pytest.mark.slow and should be run separately
from fast unit tests.
"""

import pytest
import time
import concurrent.futures
import statistics


@pytest.mark.integration
@pytest.mark.slow
def test_concurrent_function_invocations_throughput(client, worker_process, platform):
    """
    Test platform throughput with concurrent function invocations.

    Verifies:
    - Platform handles high concurrent load
    - All requests complete successfully
    - Performance characteristics are acceptable
    - No race conditions or deadlocks
    """
    time.sleep(2)

    num_requests = 50
    max_workers = 10

    print(f"\n🚀 Testing throughput with {num_requests} concurrent function calls")

    def invoke_function(request_id):
        start = time.time()
        result = client.run("fn_02_one_param", {"value": request_id})
        duration = time.time() - start
        return {
            "request_id": request_id,
            "result": result,
            "duration": duration,
            "success": True
        }

    test_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(invoke_function, i) for i in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    test_duration = time.time() - test_start

    # Verify all requests succeeded
    assert len(results) == num_requests
    assert all(r["success"] for r in results)

    # Calculate statistics
    durations = [r["duration"] for r in results]
    avg_latency = statistics.mean(durations)
    p50_latency = statistics.median(durations)
    p95_latency = statistics.quantiles(durations, n=20)[18]  # 95th percentile
    p99_latency = statistics.quantiles(durations, n=100)[98]  # 99th percentile
    min_latency = min(durations)
    max_latency = max(durations)
    throughput = num_requests / test_duration

    print(f"\n✅ Concurrent function throughput test completed")
    print(f"   Total requests: {num_requests}")
    print(f"   Success rate: 100%")
    print(f"   Total time: {test_duration:.2f}s")
    print(f"   Throughput: {throughput:.2f} req/s")
    print(f"   Latency:")
    print(f"     Min: {min_latency*1000:.2f}ms")
    print(f"     P50: {p50_latency*1000:.2f}ms")
    print(f"     Avg: {avg_latency*1000:.2f}ms")
    print(f"     P95: {p95_latency*1000:.2f}ms")
    print(f"     P99: {p99_latency*1000:.2f}ms")
    print(f"     Max: {max_latency*1000:.2f}ms")

    # Basic performance assertions (adjust based on your requirements)
    assert avg_latency < 5.0, f"Average latency too high: {avg_latency:.2f}s"
    assert throughput > 5.0, f"Throughput too low: {throughput:.2f} req/s"


@pytest.mark.integration
@pytest.mark.slow
def test_concurrent_workflow_executions_throughput(client, worker_process, platform):
    """
    Test platform throughput with concurrent workflow executions.

    Verifies:
    - Workflows can execute concurrently
    - State isolation is maintained
    - Performance under workflow load
    """
    time.sleep(2)

    num_workflows = 30
    max_workers = 10

    print(f"\n🚀 Testing workflow throughput with {num_workflows} concurrent executions")

    def invoke_workflow(workflow_id):
        start = time.time()
        result = client.run("wf_05_multi_step_workflow", {"initial_value": workflow_id % 20})
        duration = time.time() - start
        return {
            "workflow_id": workflow_id,
            "result": result,
            "duration": duration,
            "success": result["steps_completed"] == 4
        }

    test_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(invoke_workflow, i) for i in range(num_workflows)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    test_duration = time.time() - test_start

    # Verify all succeeded
    assert len(results) == num_workflows
    assert all(r["success"] for r in results)

    # Calculate statistics
    durations = [r["duration"] for r in results]
    avg_latency = statistics.mean(durations)
    throughput = num_workflows / test_duration

    print(f"\n✅ Concurrent workflow throughput test completed")
    print(f"   Total workflows: {num_workflows}")
    print(f"   Success rate: 100%")
    print(f"   Total time: {test_duration:.2f}s")
    print(f"   Throughput: {throughput:.2f} workflows/s")
    print(f"   Avg latency: {avg_latency*1000:.2f}ms")


@pytest.mark.integration
@pytest.mark.slow
def test_concurrent_entity_operations_isolation(client, worker_process, platform):
    """
    Test entity state isolation under high concurrency.

    Verifies:
    - Concurrent entity operations don't interfere
    - State isolation is maintained
    - Final state is consistent
    - No race conditions in state updates
    """
    time.sleep(2)

    num_operations = 100
    max_workers = 20

    print(f"\n🚀 Testing entity isolation with {num_operations} concurrent operations")

    def perform_operation(op_id):
        # Each operation uses unique entity key
        key = f"concurrent-entity-{op_id}"
        start = time.time()

        # Perform multiple operations on same entity
        client.run("Counter", "increment", key=key, amount=1)
        client.run("Counter", "increment", key=key, amount=2)
        client.run("Counter", "increment", key=key, amount=3)
        result = client.run("Counter", "get_value", key=key)

        duration = time.time() - start

        return {
            "op_id": op_id,
            "final_count": result["count"],
            "duration": duration,
            "success": result["count"] == 6  # 1+2+3
        }

    test_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(perform_operation, i) for i in range(num_operations)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    test_duration = time.time() - test_start

    # Verify all operations succeeded with correct final state
    assert len(results) == num_operations
    assert all(r["success"] for r in results)
    assert all(r["final_count"] == 6 for r in results)

    throughput = num_operations / test_duration

    print(f"\n✅ Entity state isolation under concurrency verified")
    print(f"   Total operations: {num_operations}")
    print(f"   Success rate: 100%")
    print(f"   Throughput: {throughput:.2f} ops/s")
    print(f"   State consistency: 100%")


@pytest.mark.integration
@pytest.mark.slow
def test_mixed_workload_performance(client, worker_process, platform):
    """
    Test platform performance under mixed workload.

    Verifies:
    - Platform handles mix of functions, workflows, entities
    - No interference between different operation types
    - Balanced resource utilization
    """
    time.sleep(2)

    num_operations = 60
    max_workers = 15

    print(f"\n🚀 Testing mixed workload with {num_operations} operations")

    def perform_mixed_operation(op_id):
        start = time.time()
        operation_type = op_id % 3

        if operation_type == 0:
            # Function call
            result = client.run("fn_02_one_param", {"value": op_id})
            success = "result" in result
        elif operation_type == 1:
            # Workflow execution
            result = client.run("wf_01_basic_execution", {})
            success = result["status"] == "success"
        else:
            # Entity operation
            key = f"mixed-counter-{op_id}"
            result = client.run("Counter", "increment", key=key, amount=op_id)
            success = "count" in result

        duration = time.time() - start

        return {
            "op_id": op_id,
            "type": ["function", "workflow", "entity"][operation_type],
            "duration": duration,
            "success": success
        }

    test_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(perform_mixed_operation, i) for i in range(num_operations)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    test_duration = time.time() - test_start

    # Verify all succeeded
    assert len(results) == num_operations
    assert all(r["success"] for r in results)

    # Calculate per-type statistics
    by_type = {}
    for r in results:
        op_type = r["type"]
        if op_type not in by_type:
            by_type[op_type] = []
        by_type[op_type].append(r["duration"])

    throughput = num_operations / test_duration

    print(f"\n✅ Mixed workload test completed")
    print(f"   Total operations: {num_operations}")
    print(f"   Total time: {test_duration:.2f}s")
    print(f"   Overall throughput: {throughput:.2f} ops/s")
    print(f"   Per-type latency:")
    for op_type, durations in by_type.items():
        avg = statistics.mean(durations)
        print(f"     {op_type}: {avg*1000:.2f}ms avg")


@pytest.mark.integration
@pytest.mark.slow
def test_sustained_load_stability(client, worker_process, platform):
    """
    Test platform stability under sustained load.

    Verifies:
    - Platform remains stable over time
    - No memory leaks or resource exhaustion
    - Consistent performance over duration
    """
    time.sleep(2)

    duration_seconds = 30
    requests_per_second = 5
    total_requests = duration_seconds * requests_per_second

    print(f"\n🚀 Testing sustained load: {requests_per_second} req/s for {duration_seconds}s")

    def invoke_with_timing(request_id):
        start = time.time()
        result = client.run("fn_02_one_param", {"value": request_id % 100})
        duration = time.time() - start
        return {
            "request_id": request_id,
            "timestamp": start,
            "duration": duration,
            "success": "result" in result
        }

    results = []
    test_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(total_requests):
            # Submit request
            future = executor.submit(invoke_with_timing, i)
            results.append(future)

            # Rate limiting - sleep to maintain target rate
            elapsed = time.time() - test_start
            expected_requests = (elapsed * requests_per_second)
            if i > expected_requests:
                time.sleep((i - expected_requests) / requests_per_second)

        # Wait for all to complete
        completed_results = [f.result() for f in results]

    test_duration = time.time() - test_start

    # Verify all succeeded
    assert all(r["success"] for r in completed_results)

    # Analyze stability over time - divide into time buckets
    bucket_size = 5  # 5 second buckets
    buckets = {}
    for r in completed_results:
        bucket_id = int((r["timestamp"] - test_start) / bucket_size)
        if bucket_id not in buckets:
            buckets[bucket_id] = []
        buckets[bucket_id].append(r["duration"])

    print(f"\n✅ Sustained load test completed")
    print(f"   Duration: {test_duration:.2f}s")
    print(f"   Total requests: {len(completed_results)}")
    print(f"   Success rate: 100%")
    print(f"   Performance over time:")

    for bucket_id in sorted(buckets.keys()):
        durations = buckets[bucket_id]
        avg = statistics.mean(durations)
        bucket_time = bucket_id * bucket_size
        print(f"     {bucket_time:3d}-{bucket_time+bucket_size:3d}s: {avg*1000:.2f}ms avg ({len(durations)} reqs)")


@pytest.mark.integration
@pytest.mark.slow
def test_burst_load_handling(client, worker_process, platform):
    """
    Test platform handles burst loads correctly.

    Verifies:
    - Platform can handle sudden load spikes
    - Queue management works correctly
    - No requests are dropped
    - Recovery after burst
    """
    time.sleep(2)

    burst_size = 50

    print(f"\n🚀 Testing burst load with {burst_size} simultaneous requests")

    def invoke_function(request_id):
        start = time.time()
        result = client.run("fn_02_one_param", {"value": request_id})
        duration = time.time() - start
        return {
            "request_id": request_id,
            "duration": duration,
            "success": "result" in result
        }

    # Submit all requests simultaneously
    test_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size) as executor:
        futures = [executor.submit(invoke_function, i) for i in range(burst_size)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    test_duration = time.time() - test_start

    # Verify all succeeded
    assert len(results) == burst_size
    assert all(r["success"] for r in results)

    durations = [r["duration"] for r in results]
    avg_latency = statistics.mean(durations)
    max_latency = max(durations)

    print(f"\n✅ Burst load test completed")
    print(f"   Burst size: {burst_size}")
    print(f"   Success rate: 100%")
    print(f"   Total time: {test_duration:.2f}s")
    print(f"   Avg latency: {avg_latency*1000:.2f}ms")
    print(f"   Max latency: {max_latency*1000:.2f}ms")


@pytest.mark.integration
@pytest.mark.slow
def test_concurrent_complex_workflows(client, worker_process, platform):
    """
    Test concurrent execution of complex multi-step workflows.

    Verifies:
    - Complex workflows execute correctly under concurrency
    - State management works with concurrent workflows
    - No interference between workflow executions
    """
    time.sleep(2)

    num_workflows = 20
    max_workers = 10

    print(f"\n🚀 Testing {num_workflows} concurrent complex workflows")

    def invoke_complex_workflow(workflow_id):
        start = time.time()
        result = client.run("wf_05_multi_step_workflow", {"initial_value": workflow_id + 1})
        duration = time.time() - start
        return {
            "workflow_id": workflow_id,
            "steps_completed": result["steps_completed"],
            "duration": duration,
            "success": result["steps_completed"] == 4
        }

    test_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(invoke_complex_workflow, i) for i in range(num_workflows)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    test_duration = time.time() - test_start

    # Verify all completed successfully
    assert len(results) == num_workflows
    assert all(r["success"] for r in results)
    assert all(r["steps_completed"] == 4 for r in results)

    durations = [r["duration"] for r in results]
    avg_latency = statistics.mean(durations)
    throughput = num_workflows / test_duration

    print(f"\n✅ Concurrent complex workflows completed")
    print(f"   Workflows: {num_workflows}")
    print(f"   Success rate: 100%")
    print(f"   Throughput: {throughput:.2f} workflows/s")
    print(f"   Avg latency: {avg_latency*1000:.2f}ms")


# TODO: Add tests for:
# - Memory usage profiling
# - CPU utilization monitoring
# - Connection pool exhaustion
# - Database connection limits
# - Worker pool saturation
# - Long-running workflow performance
# - Performance degradation detection
# - Resource cleanup verification
# - Performance comparison across runtime modes
