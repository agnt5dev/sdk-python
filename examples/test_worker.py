#!/usr/bin/env python3
"""
Test example demonstrating PyWorker and test_function from sdk-core.
"""

import sys
import os

def test_pyworker():
    """Test PyWorker functionality."""
    try:
        from agnt5 import PyWorker
        
        print("🧪 Testing PyWorker...")
        
        # Create a worker
        coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:9091")
        worker = PyWorker(coordinator_endpoint)
        
        print(f"✅ PyWorker created successfully!")
        print(f"🆔 Worker ID: {worker.worker_id()}")
        print(f"🌐 Endpoint: {worker.get_endpoint()}")
        
        return True
    except Exception as e:
        print(f"❌ PyWorker test failed: {e}")
        return False

def test_function_call():
    """Test the basic test_function."""
    try:
        from agnt5 import test_function
        
        print("🧪 Testing test_function...")
        test_function()
        print("✅ test_function executed successfully!")
        return True
    except Exception as e:
        print(f"❌ test_function failed: {e}")
        return False

def main():
    """Main entry point for the test worker."""
    print("🚀 Running AGNT5 SDK tests...")
    
    # Test basic function
    success1 = test_function_call()
    
    # Test PyWorker
    success2 = test_pyworker()
    
    if success1 and success2:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())