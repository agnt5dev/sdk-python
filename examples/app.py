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
import os
import sys
import logging

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

# 02: Entities
# Components: Counter, ConversationMemory, KeyValueStore
import ex_02_entities  # noqa: F401

# 03: Workflows
# Components: data_pipeline, order_fulfillment, etl_workflow, approval_workflow
import ex_03_workflows  # noqa: F401

# 04: LM Functions (OpenAI)
# Components: lm_generate, lm_stream, lm_summarize, lm_classify
import ex_04_lm_functions_openai  # noqa: F401

# 05: LM Functions (Anthropic)
# Components: claude_generate, claude_stream, claude_analyze, claude_translate
import ex_05_lm_functions_anthropic  # noqa: F401

# 06: Agents (Basic)
# Components: simple_agent, claude_agent, code_assistant, creative_writer, conversation_agent
import ex_06_agents  # noqa: F401

# 07: Agents with Tools
# Components: calculator_agent, weather_assistant, multi_tool_agent, research_coordinator, support_router
# Tools: calculate, get_weather, search_database, send_email
import ex_07_agents_with_tools  # noqa: F401

# 08: Human-in-the-Loop (HITL)
# Components: hitl_text_input, hitl_approval, hitl_choice, hitl_onboarding,
#             hitl_agent_questions, hitl_conditional_approval
import ex_08_hitl  # noqa: F401

# 09: Streaming
# Components: stream_text, stream_story, stream_counter, stream_progress, stream_json_chunks
import ex_09_streaming  # noqa: F401

# 10: Structured Output
# Components: analyze_code_with_dataclass, analyze_security_with_pydantic, etc.
import ex_10_structured_output  # noqa: F401

# 11: Agent Streaming
# Components: stream_agent_chat, stream_agent_with_tools, stream_agent_simple
import ex_11_agent_streaming  # noqa: F401

# 12: Workflow Streaming
# Components: research_workflow, mixed_workflow, simple_agent_workflow
import ex_12_workflow_streaming  # noqa: F401

# 13: Context Propagation
# Components: sequential_context_workflow, parallel_context_workflow, ctx_state_pipeline, etc.
import ex_13_context_propagation  # noqa: F401

# 14: Parallel Tool Calls
# Components: parallel_tool_execution, timing_comparison, sequential_tool_chain, etc.
import ex_14_parallel_tool_calls  # noqa: F401

# 15: Advanced Handoffs (if exists)
# import ex_15_advanced_handoffs  # noqa: F401

# 16: Multi-Agent Orchestration
# Components: supervisor_worker_workflow, debate_agents, consensus_agents, pipeline_agents
import ex_16_multi_agent_orchestration  # noqa: F401

from ex_06_agents import simple_assistant_agent  # noqa: F401

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
