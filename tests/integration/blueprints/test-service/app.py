#!/usr/bin/env python3
"""Test Service Worker - Integration Test Worker for AGNT5 SDK."""

import asyncio
import os
import sys
import logging
from pathlib import Path
from agnt5 import Worker

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    # Look for .env file in the worker directory
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logging.info(f"✅ Loaded environment variables from {env_file}")

        # Log which API keys are available (without exposing values)
        api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        }
        available_keys = [k for k, v in api_keys.items() if v]
        if available_keys:
            logging.info(f"🔑 Available API keys: {', '.join(available_keys)}")
        else:
            logging.warning("⚠️  No API keys found - LM integration tests will fail without API keys")
    else:
        logging.warning(f"⚠️  No .env file found at {env_file} - LM integration tests will fail without API keys")
except ImportError:
    logging.warning("⚠️  python-dotenv not installed - .env files will not be loaded")
    logging.info("   Install with: pip install python-dotenv")

# Configure logging (this will also control Rust log levels via PyO3-log)
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG to see all Rust logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SERVICE_NAME = "test-service"

async def main():
    """Main entry point for the worker."""
    logger.info("Starting %s worker...", SERVICE_NAME)

    # Configuration from environment
    coordinator_endpoint = os.getenv("AGNT5_COORDINATOR_ENDPOINT", "http://localhost:34186")

    try:
        # Get the package source directory
        package_dir = Path(__file__).parent / "src" / "agnt5_test_service"

        # Create worker with auto-registration enabled
        worker = Worker(
            service_name=SERVICE_NAME,
            service_version="1.0.0",
            coordinator_endpoint=coordinator_endpoint,
            runtime="standalone",
            auto_register=True,
            auto_register_paths=[str(package_dir)]
        )

        # Components are automatically discovered and registered
        logger.info("Worker created successfully with auto-registration from: %s", package_dir)

        # Start the worker (this is async and will block until shutdown)
        logger.info("Starting worker and registering with coordinator...")
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
