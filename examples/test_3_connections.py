#!/usr/bin/env python3
"""
Test script to send 3 consecutive connections to check coordinator logging.
"""

import os
import sys
import time

def main():
    print("🚀 Testing 3 consecutive connections...")
    
    # Configuration
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:9091")
    
    try:
        # Import PyWorker from our Rust extension
        from agnt5 import PyWorker
        
        for i in range(1, 4):
            print(f"\n--- Connection Test #{i} ---")
            worker = PyWorker(coordinator_endpoint)
            
            print(f"🆔 Worker ID: {worker.worker_id()}")
            print(f"🔍 Testing connection #{i}...")
            
            connected = worker.test_connection()
            if connected:
                print(f"✅ Connection #{i} successful!")
            else:
                print(f"❌ Connection #{i} failed!")
            
            # Small delay between connections
            if i < 3:
                time.sleep(0.5)
        
        print(f"\n🎯 All 3 connection tests completed!")
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())