#!/usr/bin/env python3
"""
Improved AGNT5 worker example with enhanced connectivity, error handling, and signal handling.

This demonstrates the improved SDK features:
- Proper Ctrl+C handling
- Connection state tracking
- Error logging and retrieval
- Exponential backoff with jitter
- Better error messages
"""

import os
import sys
import signal
import time
import threading

# Import the enhanced PyWorker with logging capabilities
from agnt5 import PyWorker
from agnt5.decorators import handler

# Sample handlers to demonstrate functionality
@handler
def greet_user(name: str) -> str:
    """Simple greeting handler."""
    return f"Hello, {name}! This is the improved worker with better connectivity."

@handler
def add_numbers(a: int, b: int) -> int:
    """Addition handler that can demonstrate error handling."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError(f"Expected integers, got {type(a)} and {type(b)}")
    return a + b

@handler
def simulate_error() -> str:
    """Handler that simulates an error for testing error logging."""
    raise RuntimeError("This is a simulated error to test error handling")

def monitor_connection_state(worker):
    """Monitor and log connection state changes."""
    last_state = None
    while worker.is_running():
        current_state = worker.get_connection_state()
        if current_state != last_state:
            print(f"🔄 Connection state changed: {last_state} -> {current_state}")
            last_state = current_state
        
        # Check for errors from Rust core
        errors = PyWorker.get_errors()
        if errors:
            print("❌ Recent errors from Rust core:")
            for error in errors[-3:]:  # Show last 3 errors
                print(f"   {error}")
            PyWorker.clear_errors()  # Clear after showing
        
        time.sleep(2)

def main():
    print("🚀 Starting improved AGNT5 worker with enhanced connectivity...")
    
    # Initialize logging first
    try:
        PyWorker.init_logging()
        print("✅ Logging initialized successfully")
    except Exception as e:
        print(f"⚠️  Could not initialize logging: {e}")
    
    # Configuration
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:9091")
    tenant_id = os.getenv("AGNT5_TENANT_ID")
    deployment_id = os.getenv("AGNT5_DEPLOYMENT_ID")
    
    print(f"🌐 Coordinator: {coordinator_endpoint}")
    if tenant_id:
        print(f"🏢 Tenant ID: {tenant_id}")
    if deployment_id:
        print(f"🚀 Deployment ID: {deployment_id}")
    
    worker = None
    monitor_thread = None
    
    def signal_handler(signum, frame):
        """Graceful shutdown handler."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        if worker and worker.is_running():
            print("📞 Stopping worker...")
            worker.stop()
            
        if monitor_thread and monitor_thread.is_alive():
            print("📊 Waiting for monitor thread...")
            monitor_thread.join(timeout=2)
            
        print("✅ Shutdown complete!")
        sys.exit(0)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create worker
        worker = PyWorker(
            coordinator_endpoint, 
            "improved-python-service", 
            "1.0.0", 
            "python",
            tenant_id,
            deployment_id
        )
        
        print(f"✅ Worker created successfully!")
        print(f"🆔 Worker ID: {worker.worker_id()}")
        print(f"🏢 Tenant ID: {worker.tenant_id()}")
        print(f"🚀 Deployment ID: {worker.deployment_id()}")
        print(f"🌐 Endpoint: {worker.get_endpoint()}")
        print(f"🔗 Initial state: {worker.get_connection_state()}")
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_connection_state, args=(worker,), daemon=True)
        monitor_thread.start()
        
        # Start the worker (this will automatically retry with exponential backoff)
        print("🚀 Starting worker with enhanced connectivity...")
        print("   ✨ Features: Signal handling, connection state tracking, error logging")
        print("   🔄 Auto-retry with exponential backoff and jitter")
        print("   💓 Heartbeat every 30 seconds")
        print("   🛑 Press Ctrl+C for graceful shutdown")
        
        worker.start()
        print("✅ Worker connected! Registration completed successfully.")
        print(f"📡 Running: {worker.is_running()}")
        print(f"🔗 State: {worker.get_connection_state()}")
        
        # Keep the worker running
        print("🔄 Worker is running with improved connectivity...")
        while worker.is_running():
            time.sleep(1)
            
    except KeyboardInterrupt:
        # This should be handled by signal handler, but just in case
        print("\n🛑 Keyboard interrupt received...")
        signal_handler(signal.SIGINT, None)
        
    except Exception as e:
        print(f"❌ Worker error: {e}")
        
        # Show any recent errors from Rust core
        errors = PyWorker.get_errors()
        if errors:
            print("📋 Recent errors from Rust core:")
            for error in errors:
                print(f"   {error}")
        
        return 1
    
    finally:
        if worker and worker.is_running():
            print("🧹 Cleaning up...")
            worker.stop()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())