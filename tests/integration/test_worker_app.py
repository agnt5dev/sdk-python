"""
AGNT5 Test Worker App

This worker imports test fixtures and registers them with the AGNT5 platform
for integration testing.

Components are registered automatically via decorators when modules are imported.
"""

import asyncio
import logging
import os
import sys

from agnt5 import Worker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import test fixtures to register components
logger.info("Loading test fixtures...")
from .fixtures import function_fixtures  # noqa: F401

logger.info("Test fixtures loaded successfully")

SERVICE_NAME = os.getenv("AGNT5_SERVICE_NAME", "agnt5-test-worker")


async def main():
    """Start the test worker."""
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:34186")

    logger.info(f"Starting {SERVICE_NAME} worker (test fixtures)...")
    logger.info(f"Coordinator: {coordinator_endpoint}")

    try:
        worker = Worker(
            service_name=SERVICE_NAME,
            service_version="1.0.0-test",
            coordinator_endpoint=coordinator_endpoint,
            runtime="standalone",
            auto_register=True,
        )

        await worker.run()

    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
