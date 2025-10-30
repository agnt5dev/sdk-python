"""AGNT5 Python SDK Integration Test Bench App"""

import asyncio
import os
import sys
import logging
from agnt5 import Worker

from agnt5_test_bench.workflows import test_workflow

SERVICE_NAME = "agnt5-test-bench"


async def main():
    """Main entry point for the test bench app."""

    logger = logging.getLogger(__name__)

    # Configuration from environment
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:34186")
    tenant_id = os.getenv("AGNT5_TENANT_ID")
    deployment_id = os.getenv("AGNT5_DEPLOYMENT_ID")

    try:
        worker = Worker(
            service_name=SERVICE_NAME,
            service_version="1.0.0",
            coordinator_endpoint=coordinator_endpoint,
            runtime="standalone",
            workflows=[test_workflow],
        )

        logger.info("Starting %s worker...", SERVICE_NAME)
        await worker.run()

    except ImportError as e:
        logger.error("Make sure to install agnt5: pip install agnt5")
        logger.error(f"Import error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
