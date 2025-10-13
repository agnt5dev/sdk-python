"""
Test Service Worker

Entry point for the integration test worker service.

This worker connects to the AGNT5 platform and registers all test components
(functions, entities, workflows) for integration testing.

Usage:
    uv run python app.py

Environment Variables:
    AGNT5_COORDINATOR_ENDPOINT - Worker Coordinator endpoint
    AGNT5_SERVICE_NAME - Service name (default: test-service)
    AGNT5_TENANT_ID - Tenant ID (default: test-tenant-001)
    AGNT5_DEPLOYMENT_ID - Deployment ID (default: test-deployment-001)
"""

import os
import sys

# Add parent directory to path so we can import components
sys.path.insert(0, os.path.dirname(__file__))

from agnt5 import Worker

# Import all test components
# This registers them with the worker
import components  # noqa: F401


def main():
    """Start the test worker service."""
    service_name = os.getenv("AGNT5_SERVICE_NAME", "test-service")
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT")

    print(f"🚀 Starting test service worker: {service_name}")
    print(f"   Coordinator: {coordinator_endpoint}")

    if not coordinator_endpoint:
        print("❌ ERROR: AGNT5_COORDINATOR_ENDPOINT environment variable not set")
        sys.exit(1)

    # Create and start worker
    worker = Worker(service_name=service_name)

    print(f"✅ Worker initialized with {len(worker._registry)} components")

    # Start the worker (blocks until shutdown)
    worker.start()


if __name__ == "__main__":
    main()
