#!/usr/bin/env python3
"""
AGNT5 worker example with function decorators.

This demonstrates using @function decorators to register handlers
that can be invoked through the AGNT5 platform.
"""

import os
import sys
import time

# Import AGNT5 SDK
from agnt5 import Worker, function


@function("add_numbers")
def add_numbers(ctx, a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"🧮 Adding {a} + {b}")
    return a + b


@function("greet_user")  
def greet_user(ctx, name: str) -> str:
    """Greet a user by name."""
    print(f"👋 Greeting user: {name}")
    return f"Hello, {name}!"


@function()  # Uses function name as handler name
def multiply(ctx, x: float, y: float) -> float:
    """Multiply two numbers."""
    print(f"✖️ Multiplying {x} * {y}")
    return x * y


def main():
    print("🚀 Starting AGNT5 function worker...")
    
    # Configuration
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:9091")
    service_name = "my-python-service"
    
    print(f"🌐 Coordinator: {coordinator_endpoint}")
    print(f"📦 Service: {service_name}")
    
    try:
        # Create worker - this will automatically discover @function decorators
        worker = Worker(service_name, coordinator_endpoint=coordinator_endpoint)
        
        print(f"✅ Worker created successfully!")
        print(f"🆔 Worker ID: {worker.worker_id}")
        
        # Start the worker - this will register all @function handlers
        print("🚀 Starting worker and registering with coordinator...")
        worker.start()
        print("✅ Worker connected! Functions registered with coordinator.")
        
        try:
            # Keep the worker running
            print("🔄 Worker is running. Press Ctrl+C to stop...")
            print("📋 Registered functions: add_numbers, greet_user, multiply")
            print()
            print("Test with:")
            print('grpcurl -plaintext -d \'{"serviceName": "my-python-service", "handlerName": "add_numbers", "inputData": "eyJhIjogNSwgImIiOiAzfQ=="}\' localhost:34182 api.v1.GatewayService/InvokeHandler')
            print()
            
            while worker.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping worker...")
            worker.stop()
            print("✅ Worker stopped!")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to build the Rust extension first")
        return 1
    except Exception as e:
        print(f"❌ Worker error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())