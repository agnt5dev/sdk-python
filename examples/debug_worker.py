#!/usr/bin/env python3
"""
Debug version of the AGNT5 function worker with breakpoints.
"""

import os
import sys
import time
import pdb

# Import AGNT5 SDK
from agnt5 import Worker, function


@function("add_numbers")
def add_numbers(ctx, a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"🧮 Adding {a} + {b}")
    result = a + b
    print(f"Result: {result}")
    return result


@function("greet_user")  
def greet_user(ctx, name: str) -> str:
    """Greet a user by name."""
    print(f"👋 Greeting user: {name}")
    return f"Hello, {name}!"


def main():
    print("🐛 Starting AGNT5 function worker with debugging...")
    
    # Configuration
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:9091")
    service_name = "debug-python-service"
    
    print(f"🌐 Coordinator: {coordinator_endpoint}")
    print(f"📦 Service: {service_name}")
    print("💡 Use pdb commands: n (next), c (continue), l (list), p <var> (print)")
    
    try:
        # Create worker
        worker = Worker(service_name, coordinator_endpoint=coordinator_endpoint)
        
        print(f"✅ Worker created successfully!")
        print(f"🆔 Worker ID: {worker.worker_id}")
        print(f"🏢 Tenant ID: {worker.tenant_id}")
        print(f"🚀 Deployment ID: {worker.deployment_id}")
        
        # Start the worker
        print("🚀 Starting worker and registering with coordinator...")
        worker.start()
        print("✅ Worker connected! Functions registered with coordinator.")
        
        try:
            # Keep the worker running
            print("🔄 Worker is running. Press Ctrl+C to stop...")
            print("📋 Registered functions: add_numbers, greet_user")
            print()
            print("Test with:")
            print(f'grpcurl -plaintext -d \'{{"serviceName": "{service_name}", "handlerName": "add_numbers", "inputData": "eyJhIjogNSwgImIiOiAzfQ=="}}\' localhost:34182 api.v1.GatewayService/InvokeHandler')
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