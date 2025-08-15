#!/usr/bin/env python3
"""
Simple AGNT5 worker example - creates a worker and shows basic functionality.

This demonstrates the minimal SDK functionality using the simplified PyWorker.
"""

import os
import sys

def main():
    print("🚀 Starting simple AGNT5 worker...")
    
    # Configuration
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:9091")
    
    print(f"🌐 Coordinator: {coordinator_endpoint}")
    
    try:
        # Import PyWorker from our Rust extension
        from agnt5 import PyWorker
        
        # Create worker
        worker = PyWorker(coordinator_endpoint, "simple-python-service", "1.0.0", "python")
        
        print(f"✅ Worker created successfully!")
        print(f"🆔 Worker ID: {worker.worker_id()}")
        print(f"🌐 Endpoint: {worker.get_endpoint()}")
        
        # Start the worker
        print("🚀 Starting worker and registering with coordinator...")
        worker.start()
        print("✅ Worker connected! Registration completed successfully.")
        print(f"📡 Running: {worker.is_running()}")
        
        try:
            # Keep the worker running
            print("🔄 Worker is running. Press Ctrl+C to stop...")
            import time
            while worker.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping worker...")
            worker.stop()
            print("✅ Worker stopped!")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to build the Rust extension first with: cargo build")
        return 1
    except Exception as e:
        print(f"❌ Worker error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())