"""
AGNT5 Examples Worker

This worker imports all example modules and registers their components
with the AGNT5 platform. Used for:
- Local development (just platform standalone python)
- Integration testing (pytest tests/integration/)
- CI pipelines

Components are registered automatically via decorators when modules are imported.
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load .env file from the examples directory
load_dotenv()

from agnt5 import Worker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import example modules to trigger component registration
# Each module uses @function, @workflow, @entity decorators
# that auto-register components when the module is imported

# 01: Basic Functions
# Components: greet, add, process_data, failing_function
import ex_01_functions  # noqa: F401
import ex_03_workflows  # noqa: F401
import ex_04_lm_functions_openai  # noqa: F401
import ex_05_lm_functions_anthropic  # noqa: F401
import ex_06_agents  # noqa: F401
import ex_07_agents_with_tools  # noqa: F401
import ex_08_hitl  # noqa: F401
import ex_09_streaming  # noqa: F401
import ex_10_structured_output  # noqa: F401
import ex_11_agent_streaming  # noqa: F401
import ex_12_workflow_streaming  # noqa: F401
import ex_13_context_propagation  # noqa: F401
import ex_14_parallel_tool_calls  # noqa: F401
import ex_15_advanced_handoffs  # noqa: F401
import ex_16_multi_agent_orchestration  # noqa: F401

SERVICE_NAME = os.getenv("AGNT5_SERVICE_NAME", "agnt5-examples")


async def main():
    """Start the examples worker."""
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:34186")

    logger.info(f"Starting {SERVICE_NAME} worker...")
    logger.info(f"Coordinator: {coordinator_endpoint}")

    try:
        worker = Worker(
            service_name=SERVICE_NAME,
            service_version="1.0.0",
            coordinator_endpoint=coordinator_endpoint,
            runtime="standalone",
            auto_register=True,
        )

        await worker.run()

    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
