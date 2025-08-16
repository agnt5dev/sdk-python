#!/usr/bin/env python3
"""
AGNT5 worker example with tenant and deployment isolation.

This demonstrates the enhanced SDK functionality with multi-tenancy.
"""

import os
import sys

def main():
    print("🚀 Starting tenant-aware AGNT5 worker...")
    
    # Configuration
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:9091")
    tenant_id = os.getenv("AGNT5_TENANT_ID", "customer-123")
    deployment_id = os.getenv("AGNT5_DEPLOYMENT_ID", "prod-west")
    
    print(f"🌐 Coordinator: {coordinator_endpoint}")
    print(f"🏢 Tenant ID: {tenant_id}")
    print(f"🚀 Deployment ID: {deployment_id}")
    
    try:
        # Import PyWorker from our Rust extension
        from agnt5 import PyWorker
        
        # Create worker with explicit tenant and deployment
        worker = PyWorker(
            coordinator_endpoint, 
            "tenant-python-service", 
            "1.0.0", 
            "python",
            tenant_id=tenant_id,
            deployment_id=deployment_id
        )
        
        print(f"✅ Tenant-aware worker created successfully!")
        print(f"🆔 Worker ID: {worker.worker_id()}")
        print(f"🌐 Endpoint: {worker.get_endpoint()}")
        
        # Start the worker
        print("🚀 Starting worker and registering with coordinator...")
        print(f"   💡 Worker will register with tenant '{tenant_id}' and deployment '{deployment_id}'")
        print("   💓 Heartbeat will be sent every 30 seconds (configurable via AGNT5_HEARTBEAT_INTERVAL_SECS)")
        
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
            print("\n🛑 Stopping worker gracefully...")
            worker.stop()
            print("✅ Worker stopped!")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the AGNT5 Python SDK is built and installed")
        return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())