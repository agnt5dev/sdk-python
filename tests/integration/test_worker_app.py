"""
AGNT5 Test Worker App

This worker imports test fixtures (instead of examples) and registers their
components with the AGNT5 platform for integration testing.

This allows integration tests to use comprehensive test fixtures while keeping
examples clean and pedagogical.

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

# Import all test fixture modules to trigger component registration
# This executes all @function, @workflow, @entity decorators
logger.info("Loading test fixtures...")
from fixtures import function_fixtures  # noqa: F401
from fixtures import entity_fixtures  # noqa: F401
from fixtures import workflow_fixtures  # noqa: F401
from fixtures import agent_fixtures  # noqa: F401

logger.info("Test fixtures loaded successfully")

# Also import example modules for backward compatibility with existing tests
# that reference example function names like "greet", "add", etc.
# We'll gradually migrate these to fixture-based tests
examples_dir = os.path.join(os.path.dirname(__file__), "../../examples")
sys.path.insert(0, examples_dir)

try:
    logger.info("Loading examples for backward compatibility...")
    import ex_01_functions  # noqa: F401
    import ex_02_entities  # noqa: F401
    import ex_03_workflows  # noqa: F401
    # Add more as needed for backward compatibility
    logger.info("Examples loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import examples (this is OK for fixture-only tests): {e}")

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
