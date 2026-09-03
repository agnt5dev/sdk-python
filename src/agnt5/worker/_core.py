"""Worker implementation for AGNT5 SDK.

Supports functions, entities, workflows, agents, and tools.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Literal

from .. import _sentry
from .._serialization import serialize_to_str
from .._telemetry import (
    ensure_root_otel_handler,
    init_sdk_telemetry,
    run_scope,
    setup_module_logger,
)
from ..function import FunctionRegistry
from ..scorer import (
    BUILTIN_DETERMINISTIC_SCORER_NAMES,
    ScorerRegistry,
    all_builtin_judge_scorers,
    get_builtin_judge_scorer_config,
    register_builtin_scorer_handlers,
)
from ._executors import ExecutorMixin
from ._prompt_executor import (
    PROMPT_EXECUTOR_COMPONENT_NAME,
    PROMPT_EXECUTOR_INPUT_SCHEMA,
    PROMPT_EXECUTOR_METADATA,
    PROMPT_EXECUTOR_OUTPUT_SCHEMA,
    is_prompt_executor_component,
    prompt_executor_config,
)

logger = setup_module_logger(__name__)

_BUILTIN_COMPONENT_SOURCE = "agnt5_builtin"


def _run_id_of(invocation_id: str) -> str:
    """Run id for an invocation id, dropping any ``:suffix``."""
    return invocation_id.split(":", 1)[0]


def _set_positive_int_env(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    os.environ[name] = str(value)


def _configure_worker_environment(
    *,
    coordinator_endpoint: str | None = None,
    project_id: str | None = None,
    deployment_id: str | None = None,
    worker_mode: Literal["push", "pull"] | None = None,
    parked_polling: bool | None = None,
    min_slots: int | None = None,
    max_slots: int | None = None,
    claim_timeout_ms: int | None = None,
) -> None:
    if coordinator_endpoint:
        os.environ["AGNT5_COORDINATOR_ENDPOINT"] = coordinator_endpoint
    if project_id:
        os.environ["AGNT5_PROJECT_ID"] = project_id
    if deployment_id:
        os.environ["AGNT5_DEPLOYMENT_ID"] = deployment_id

    if worker_mode not in (None, "push", "pull"):
        raise ValueError("worker_mode must be 'push' or 'pull'")
    if worker_mode == "pull" and parked_polling is False:
        raise ValueError("pull workers always use long polling")
    resolved_mode = worker_mode or ("pull" if parked_polling else None)
    if resolved_mode:
        os.environ["AGNT5_WORKER_MODE"] = resolved_mode

    _set_positive_int_env("AGNT5_MIN_SLOTS", min_slots)
    _set_positive_int_env("AGNT5_MAX_SLOTS", max_slots)
    _set_positive_int_env("AGNT5_CLAIM_TIMEOUT_MS", claim_timeout_ms)


def _is_system_component(component: Any) -> bool:
    """Return True for platform-injected components hidden from user summaries."""
    metadata = getattr(component, "metadata", {}) or {}
    return metadata.get("source") == _BUILTIN_COMPONENT_SOURCE


class Worker(ExecutorMixin):
    """AGNT5 Worker for registering and executing the components.

    The Worker class manages the lifecycle of your service, including:
    - Registration with the AGNT5 platform
    - Automatic discovery of components
    - Message handling and execution
    - Health monitoring

    Example:
        ```python
        from agnt5 import Worker, function

        @function
        async def process_data(ctx: Context, data: str) -> dict:
            return {"result": data.upper()}

        async def main():
            worker = Worker(
                service_name="data-processor",
                service_version="1.0.0",
                coordinator_endpoint="http://localhost:34186"
            )
            await worker.run()

        if __name__ == "__main__":
            asyncio.run(main())
        ```
    """

    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        coordinator_endpoint: str | None = None,
        runtime: str = "standalone",
        metadata: dict[str, str] | None = None,
        functions: list | None = None,
        workflows: list | None = None,
        entities: list | None = None,
        agents: list | None = None,
        tools: list | None = None,
        scorers: list | None = None,
        auto_register: bool = False,
        auto_register_paths: list[str] | None = None,
        pyproject_path: str | None = None,
        max_concurrency: int | None = None,
        project_id: str | None = None,
        deployment_id: str | None = None,
        worker_mode: Literal["push", "pull"] | None = None,
        parked_polling: bool | None = None,
        min_slots: int | None = None,
        max_slots: int | None = None,
        claim_timeout_ms: int | None = None,
        activation_artifact_sha256: str | None = None,
    ):
        """Initialize a new Worker with explicit or automatic component registration.

        The Worker supports two registration modes:

        **Explicit Mode:**
        - Register workflows/agents explicitly, their dependencies are auto-included
        - Optionally register standalone functions/tools for direct API invocation

        **Auto-Registration Mode (development):**
        - Automatically discovers all decorated components in source paths
        - Reads source paths from pyproject.toml or uses explicit paths
        - No need to maintain import lists

        Args:
            service_name: Unique name for this service
            service_version: Version string (semantic versioning recommended)
            coordinator_endpoint: Coordinator endpoint URL (default: from env AGNT5_COORDINATOR_ENDPOINT)
            runtime: Runtime type - "standalone", "docker", "kubernetes", etc.
            metadata: Optional service-level metadata
            functions: List of @function decorated handlers (explicit mode)
            workflows: List of @workflow decorated handlers (explicit mode)
            entities: List of Entity classes (explicit mode)
            agents: List of Agent instances (explicit mode)
            tools: List of Tool instances (explicit mode)
            scorers: List of @scorer decorated handlers (explicit mode)
            auto_register: Enable automatic component discovery (default: False)
            auto_register_paths: Explicit source paths to scan (overrides pyproject.toml discovery)
            pyproject_path: Path to pyproject.toml (default: current directory)
            max_concurrency: Max in-flight invocations this worker serves. Sets the
                local pool size and the coordinator's per-priority headroom denominator.
                Defaults to the AGNT5_MAX_CONCURRENCY env var, then 100. Raise it for
                async/IO-bound LLM workflows; lower it for CPU-bound work.
            project_id: Project routing key. Sets AGNT5_PROJECT_ID for parked polling
                and defaults service metadata `project_id` when not already provided.
            deployment_id: Deployment routing key. Sets AGNT5_DEPLOYMENT_ID for parked
                polling and defaults service metadata `deployment_id` when not already provided.
            worker_mode: Assignment mode. Use "pull" for worker-side polling or "push"
                for stream dispatch. Defaults to AGNT5_WORKER_MODE, then push.
            parked_polling: Deprecated compatibility alias. True implies
                worker_mode="pull"; pull workers always use long polling.
            min_slots: Minimum parked poll slots to keep open.
            max_slots: Maximum parked poll slots.
            claim_timeout_ms: Lease duration for claimed jobs, in milliseconds.
            activation_artifact_sha256: Immutable deployed artifact SHA-256 used
                in durable activation identity. Managed runtimes normally inject it.
        """
        self.service_name = service_name
        self.service_version = service_version
        self.coordinator_endpoint = coordinator_endpoint
        self.runtime = runtime

        # Initialize telemetry before autodiscovery and user startup logs so
        # `agnt5 dev` and headless worker boot logs reach OTEL.
        if init_sdk_telemetry(service_name, service_version):
            ensure_root_otel_handler()

        # Initialize metadata with user-provided values
        self.metadata = dict(metadata or {})
        activation_artifact_sha256 = (
            activation_artifact_sha256
            or self.metadata.get("activation_artifact_sha256")
            or os.getenv("AGNT5_ACTIVATION_ARTIFACT_SHA256")
        )
        if activation_artifact_sha256:
            self.metadata.setdefault("activation_artifact_sha256", activation_artifact_sha256)
            os.environ["AGNT5_ACTIVATION_ARTIFACT_SHA256"] = activation_artifact_sha256
        project_id = project_id or self.metadata.get("project_id")
        deployment_id = deployment_id or self.metadata.get("deployment_id")
        _configure_worker_environment(
            coordinator_endpoint=coordinator_endpoint,
            project_id=project_id,
            deployment_id=deployment_id,
            worker_mode=worker_mode,
            parked_polling=parked_polling,
            min_slots=min_slots,
            max_slots=max_slots,
            claim_timeout_ms=claim_timeout_ms,
        )
        if project_id and "project_id" not in self.metadata:
            self.metadata["project_id"] = project_id
        if deployment_id and "deployment_id" not in self.metadata:
            self.metadata["deployment_id"] = deployment_id

        # Auto-populate identity scopes from environment if not provided.
        # `project_id` carries the project routing key. `tenant_id` is
        # reserved for the customer sub-tenant on the dispatch path and is
        # not stamped from the worker's static config — it arrives per
        # request via dispatch metadata.
        if "project_id" not in self.metadata:
            project_id = os.getenv("AGNT5_PROJECT_ID")
            if project_id:
                self.metadata["project_id"] = project_id

        if "deployment_id" not in self.metadata:
            deployment_id = os.getenv("AGNT5_DEPLOYMENT_ID")
            if deployment_id:
                self.metadata["deployment_id"] = deployment_id

        # Import Rust worker
        try:
            from .. import _core as rust_core
            from .._core import PyComponentInfo, PyTriggerSpec, PyWorker, PyWorkerConfig

            self._PyWorker = PyWorker
            self._PyWorkerConfig = PyWorkerConfig
            self._PyComponentInfo = PyComponentInfo
            self._PyTriggerSpec = PyTriggerSpec
            self._PyActivationClient = getattr(rust_core, "PyActivationClient", None)
        except ImportError as e:
            _sentry.capture_exception(
                e,
                context={
                    "service_name": service_name,
                    "service_version": service_version,
                    "error_location": "Worker.__init__",
                    "error_phase": "rust_core_import",
                },
                tags={
                    "sdk_error": "true",
                    "error_type": "import_error",
                    "component": "rust_core",
                },
                level="error",
            )
            raise ImportError(
                f"Failed to import Rust core worker: {e}. "
                "Make sure agnt5 is properly installed with: pip install agnt5"
            ) from e

        # Create Rust worker
        self._rust_config = self._PyWorkerConfig(
            service_name=service_name,
            service_version=service_version,
            service_type=runtime,
            max_concurrency=max_concurrency,
        )
        self._rust_worker = self._PyWorker(self._rust_config)
        self._activation_client = None
        if self._PyActivationClient is not None:
            from ..activation import ActivationClient, NativeActivationTransport

            endpoint = os.getenv("AGNT5_ENGINE_URL") or coordinator_endpoint
            native_activation_client = self._PyActivationClient(endpoint, self._rust_worker)
            self._activation_client = ActivationClient(
                NativeActivationTransport(native_activation_client)
            )

        # ChatBot registry: maps agent name -> ChatBot instance
        # Populated when ChatBot instances are passed in the agents list
        self._chatbots: dict[str, Any] = {}

        # In-flight invocation tasks keyed by run_id, for cooperative
        # cancellation. Populated/cleared by the tracked-invocation wrapper on
        # the event loop thread; read on the same thread via _on_cancel's
        # call_soon_threadsafe callback.
        self._inflight: dict[str, asyncio.Task] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

        # Create entity state adapter with canonical project identity. The Rust
        # state path still uses the legacy `tenant_id` field name internally.
        from .._core import EntityStateManager as RustEntityStateManager
        from .._state_adapter import StateAdapter as EntityStateAdapter

        project_id = self.metadata.get("project_id", "")
        rust_core = RustEntityStateManager(tenant_id=project_id)
        self._entity_state_adapter = EntityStateAdapter(rust_core=rust_core)

        # Create CheckpointClient for step-level memoization
        try:
            from ..checkpoint import CheckpointClient

            self._checkpoint_client = CheckpointClient()
        except Exception as e:
            logger.warning(f"Failed to create CheckpointClient (memoization disabled): {e}")
            self._checkpoint_client = None

        # Initialize Sentry
        from ..version import _get_version

        sdk_version = _get_version()
        sentry_enabled = _sentry.initialize_sentry(
            service_name=service_name,
            service_version=service_version,
            sdk_version=sdk_version,
        )
        if sentry_enabled:
            _sentry.set_context(
                "service",
                {
                    "name": service_name,
                    "version": service_version,
                    "runtime": runtime,
                },
            )
        else:
            logger.debug("SDK telemetry not enabled")

        # Third-party framework capture (openai, google-adk) — attach before
        # component discovery imports user modules so class-level patches are
        # in place ahead of any client construction.
        try:
            from ..integrations import auto_enable as _capture_auto_enable

            _capture_auto_enable()
        except Exception:
            logger.warning("third-party capture auto-enable failed", exc_info=True)

        # Component registration
        if auto_register:
            if any([functions, workflows, entities, agents, tools, scorers]):
                logger.warning(
                    "auto_register=True ignores explicit functions/workflows/entities/agents/tools/scorers parameters. "
                    "Remove explicit params or set auto_register=False to use explicit registration."
                )

            if auto_register_paths:
                source_paths = auto_register_paths
            else:
                source_paths = self._discover_source_paths(pyproject_path)

            self._auto_discover_components(source_paths)
        else:
            # Separate ChatBot instances from plain agents.
            # ChatBots wrap agents and register as agent components, but
            # we need to track them separately for webhook dispatch.
            raw_agents = list(agents or [])
            resolved_agents = []
            for a in raw_agents:
                from ..chat import ChatBot

                if isinstance(a, ChatBot):
                    self._chatbots[a.name] = a
                    resolved_agents.append(a.agent)
                    logger.debug(f"Registered ChatBot for agent '{a.name}'")
                else:
                    resolved_agents.append(a)

            self._explicit_components = {
                "functions": list(functions or []),
                "workflows": list(workflows or []),
                "entities": list(entities or []),
                "agents": resolved_agents,
                "tools": list(tools or []),
                "scorers": list(scorers or []),
            }

            total_explicit = sum(len(v) for v in self._explicit_components.values())
            logger.debug(
                f"Worker initialized: {service_name} v{service_version} (runtime: {runtime}), "
                f"{total_explicit} components registered"
            )

    def register_components(
        self,
        functions: list | None = None,
        workflows: list | None = None,
        entities: list | None = None,
        agents: list | None = None,
        tools: list | None = None,
        scorers: list | None = None,
    ) -> None:
        """Register additional components after Worker initialization.

        This method allows incremental registration of components after the Worker
        has been created. Useful for conditional or dynamic component registration.

        Args:
            functions: List of functions decorated with @function
            workflows: List of workflows decorated with @workflow
            entities: List of entity classes
            agents: List of agent instances
            tools: List of tool instances
            scorers: List of scorers decorated with @scorer
        """
        if functions:
            self._explicit_components["functions"].extend(functions)
            logger.debug(f"Registered {len(functions)} functions")

        if workflows:
            self._explicit_components["workflows"].extend(workflows)
            logger.debug(f"Registered {len(workflows)} workflows")

        if entities:
            self._explicit_components["entities"].extend(entities)
            logger.debug(f"Registered {len(entities)} entities")

        if agents:
            self._explicit_components["agents"].extend(agents)
            logger.debug(f"Registered {len(agents)} agents")

        if tools:
            self._explicit_components["tools"].extend(tools)
            logger.debug(f"Registered {len(tools)} tools")

        if scorers:
            self._explicit_components["scorers"].extend(scorers)
            logger.debug(f"Registered {len(scorers)} scorers")

        total = sum(len(v) for v in self._explicit_components.values())
        logger.info(f"Total components now registered: {total}")

    def _discover_source_paths(self, pyproject_path: str | None = None) -> list[str]:
        """Discover source paths from pyproject.toml.

        Reads pyproject.toml to find package source directories using:
        - Hatch: [tool.hatch.build.targets.wheel] packages
        - Maturin: [tool.maturin] python-source
        - Fallback: ["src"] if not found

        Args:
            pyproject_path: Path to pyproject.toml (default: current directory)

        Returns:
            List of directory paths to scan (e.g., ["src/agnt5_benchmark"])
        """
        try:
            import tomllib
        except ImportError:
            logger.error("tomllib not available (Python 3.11+ required for auto-registration)")
            return ["src"]

        if pyproject_path:
            pyproject_file = Path(pyproject_path)
        else:
            pyproject_file = Path.cwd() / "pyproject.toml"

        if not pyproject_file.exists():
            logger.warning(
                f"pyproject.toml not found at {pyproject_file}, defaulting to 'src/' directory"
            )
            return ["src"]

        try:
            with open(pyproject_file, "rb") as f:
                import tomllib

                config = tomllib.load(f)
        except Exception as e:
            logger.error(f"Failed to parse pyproject.toml: {e}")
            return ["src"]

        source_paths = []

        # Try Hatch configuration
        if "tool" in config and "hatch" in config["tool"]:
            hatch_config = config["tool"]["hatch"]
            if "build" in hatch_config and "targets" in hatch_config["build"]:
                wheel_config = hatch_config["build"]["targets"].get("wheel", {})
                packages = wheel_config.get("packages", [])
                source_paths.extend(packages)

        # Try Maturin configuration
        if not source_paths and "tool" in config and "maturin" in config["tool"]:
            maturin_config = config["tool"]["maturin"]
            python_source = maturin_config.get("python-source")
            if python_source:
                source_paths.append(python_source)

        if not source_paths:
            source_paths = ["src"]

        return source_paths

    def _auto_discover_components(self, source_paths: list[str]) -> None:
        """Auto-discover components by importing all Python files in source paths.

        Args:
            source_paths: List of directory paths to scan
        """
        import importlib.util
        import sys

        for source_path in source_paths:
            path = Path(source_path)

            if not path.exists():
                logger.warning(f"Source path does not exist: {source_path}")
                continue

            for py_file in path.rglob("*.py"):
                if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
                    continue

                relative_path = py_file.relative_to(path.parent)
                module_parts = list(relative_path.parts[:-1])
                module_parts.append(relative_path.stem)
                module_name = ".".join(module_parts)

                try:
                    if module_name in sys.modules:
                        logger.debug(f"Module already imported: {module_name}")
                    else:
                        spec = importlib.util.spec_from_file_location(module_name, py_file)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)
                            logger.debug(f"Auto-imported: {module_name}")
                except Exception as e:
                    logger.warning(f"Failed to import {module_name}: {e}")
                    _sentry.capture_exception(
                        e,
                        context={
                            "service_name": self.service_name,
                            "module_name": module_name,
                            "source_path": str(py_file),
                            "error_location": "_auto_discover_components",
                        },
                        tags={
                            "sdk_error": "true",
                            "error_type": "auto_registration_failure",
                        },
                        level="warning",
                    )

        # Collect components from registries
        from ..agent import AgentRegistry
        from ..tool import ToolRegistry
        from ..workflow import WorkflowRegistry

        functions = [cfg.handler for cfg in FunctionRegistry.all().values()]
        workflows = [cfg.handler for cfg in WorkflowRegistry.all().values()]
        entities: list = []  # Entity API removed in 0.4.0
        agents = list(AgentRegistry.all().values())
        tools = list(ToolRegistry.all().values())
        scorers = [cfg.handler for cfg in ScorerRegistry.all().values()]

        self._explicit_components = {
            "functions": functions,
            "workflows": workflows,
            "entities": entities,
            "agents": agents,
            "tools": tools,
            "scorers": scorers,
        }

        logger.info(
            f"Auto-discovered components: {len(functions)} functions, {len(entities)} entities, "
            f"{len(workflows)} workflows, {len(agents)} agents, {len(tools)} tools, {len(scorers)} scorers"
        )

    def _serialize_schema(self, schema: Any) -> str | None:
        """Serialize a schema to JSON string, returning None if empty."""
        return serialize_to_str(schema) if schema else None

    def _create_component_info(
        self,
        name: str,
        component_type: str,
        metadata: dict | None = None,
        config: dict | None = None,
        input_schema: Any = None,
        output_schema: Any = None,
        definition: Any = None,
        triggers: list | None = None,
    ) -> Any:
        """Create a PyComponentInfo with serialized schemas."""
        py_triggers = [
            self._PyTriggerSpec(
                trigger_id=getattr(trigger, "trigger_id", ""),
                trigger_type=getattr(trigger, "trigger_type", ""),
                event_name=getattr(trigger, "event_name", ""),
                filter_expression=getattr(trigger, "filter_expression", ""),
                input_mapping=getattr(trigger, "input_mapping", ""),
                batch_window_ms=getattr(trigger, "batch_window_ms", 0),
                delay_expression=getattr(trigger, "delay_expression", ""),
            )
            for trigger in (triggers or [])
        ]
        return self._PyComponentInfo(
            name=name,
            component_type=component_type,
            metadata=metadata or {},
            config=config or {},
            input_schema=self._serialize_schema(input_schema),
            output_schema=self._serialize_schema(output_schema),
            definition=self._serialize_schema(definition),
            triggers=py_triggers,
        )

    def _discover_components(self) -> list:
        """Discover explicit components (functions, entities, workflows, agents, tools).

        Returns:
            List of PyComponentInfo instances for all components
        """
        components = []

        # Process functions — match by handler identity, not by name,
        # so @function(name="custom") works correctly.
        for config in FunctionRegistry.all().values():

            config_dict = {}
            if config.retries:
                config_dict.update(
                    {
                        "max_attempts": str(config.retries.max_attempts),
                        "initial_interval_ms": str(config.retries.initial_interval_ms),
                        "max_interval_ms": str(config.retries.max_interval_ms),
                    }
                )
            if config.backoff:
                config_dict.update(
                    {
                        "backoff_type": config.backoff.type.value,
                        "backoff_multiplier": str(config.backoff.multiplier),
                    }
                )

            components.append(
                self._create_component_info(
                    name=config.name,
                    component_type="function",
                    metadata=config.metadata,
                    config=config_dict,
                    input_schema=config.input_schema,
                    output_schema=config.output_schema,
                )
            )

        if not FunctionRegistry.get(PROMPT_EXECUTOR_COMPONENT_NAME):
            components.append(
                self._create_component_info(
                    name=PROMPT_EXECUTOR_COMPONENT_NAME,
                    component_type="function",
                    metadata=PROMPT_EXECUTOR_METADATA,
                    config={},
                    input_schema=PROMPT_EXECUTOR_INPUT_SCHEMA,
                    output_schema=PROMPT_EXECUTOR_OUTPUT_SCHEMA,
                )
            )

        # Entity API removed in 0.4.0 - entities list is always empty

        from ..workflow import WorkflowRegistry

        # Process workflows — iterate registry directly,
        # so @workflow(name="custom") works correctly.
        for config in WorkflowRegistry.all().values():

            components.append(
                self._create_component_info(
                    name=config.name,
                    component_type="workflow",
                    metadata=config.metadata,
                    config={},
                    input_schema=config.input_schema,
                    output_schema=config.output_schema,
                    triggers=config.triggers,
                )
            )

        # Process agents
        for agent in self._explicit_components["agents"]:
            # Build agent definition with tool schemas
            tool_schemas = []
            for tool_name, tool in agent.tools.items():
                tool_schemas.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                )

            definition = {
                "instructions": agent.instructions,
                "model": agent.model,
                "max_iterations": agent.max_iterations,
                "tools": tool_schemas,
                "handoffs": [h.agent.name for h in agent.handoffs] if agent.handoffs else [],
            }

            components.append(
                self._create_component_info(
                    name=agent.name,
                    component_type="agent",
                    metadata={},
                    config={
                        "model": agent.model,
                        "max_iterations": str(agent.max_iterations),
                    },
                    definition=definition,
                )
            )

        # Process tools
        for tool in self._explicit_components["tools"]:
            components.append(
                self._create_component_info(
                    name=tool.name,
                    component_type="tool",
                    metadata={},
                    config={
                        "confirmation": str(tool.confirmation),
                    },
                    input_schema=tool.input_schema,
                    output_schema=tool.output_schema,
                )
            )

        registered_scorer_names: set[str] = set()

        # Process explicit/custom scorers
        for scorer_handler in self._explicit_components.get("scorers", []):
            scorer_name = getattr(scorer_handler, "_scorer_name", scorer_handler.__name__)
            config = ScorerRegistry.get(scorer_name)
            if not config:
                logger.warning(f"Scorer '{scorer_name}' not found in ScorerRegistry")
                continue

            components.append(
                self._create_component_info(
                    name=config.name,
                    component_type="scorer",
                    metadata={
                        "scope": config.scope,
                        "description": config.description,
                    },
                    config={},
                    input_schema=None,  # Scorers use standardized ScorerRequest
                    output_schema=None,  # Scorers use standardized ScorerResult
                )
            )
            registered_scorer_names.add(config.name)

        # AGNT5-owned deterministic built-ins are routable worker capabilities,
        # but native SDK-core intercepts them before Python component dispatch.
        for scorer_name in BUILTIN_DETERMINISTIC_SCORER_NAMES:
            if scorer_name in registered_scorer_names:
                continue
            components.append(
                self._create_component_info(
                    name=scorer_name,
                    component_type="scorer",
                    metadata={"source": _BUILTIN_COMPONENT_SOURCE},
                    config={},
                    input_schema=None,
                    output_schema=None,
                )
            )
            registered_scorer_names.add(scorer_name)

        # AGNT5-owned judge built-ins fall through to their Python handlers.
        for scorer_name, config in all_builtin_judge_scorers().items():
            if scorer_name in registered_scorer_names:
                continue
            components.append(
                self._create_component_info(
                    name=config.name,
                    component_type="scorer",
                    metadata={
                        "scope": config.scope,
                        "description": config.description,
                        "source": _BUILTIN_COMPONENT_SOURCE,
                    },
                    config={},
                    input_schema=None,
                    output_schema=None,
                )
            )
            registered_scorer_names.add(config.name)

        return components

    def _create_message_handler(self) -> Any:
        """Create the message handler that will be called by Rust worker.

        Handles function, entity, and workflow components.
        """

        def resolve_coro(request: Any) -> Any:
            """Resolve the handler coroutine for a request (no task tracking)."""
            component_name = request.component_name
            component_type = request.component_type
            input_data = request.input_data

            logger.debug(
                f"Handling {component_type} request: {component_name}, "
                f"input size: {len(input_data)} bytes"
            )

            # Functions
            if component_type == "function":
                function_config = FunctionRegistry.get(component_name)
                if function_config:
                    return self._execute_function(function_config, input_data, request)
                if is_prompt_executor_component(component_name):
                    return self._execute_function(
                        prompt_executor_config(component_name),
                        input_data,
                        request,
                    )

            # Entities - removed in 0.4.0
            elif component_type == "entity":
                error_msg = (
                    f"Entity API was removed in 0.4.0. Entity '{component_name}' not supported."
                )
                logger.error(error_msg)

            # Workflows
            elif component_type == "workflow":
                from ..workflow import WorkflowRegistry

                workflow_config = WorkflowRegistry.get(component_name)
                if workflow_config:
                    return self._execute_workflow(workflow_config, input_data, request)

            # Tools
            elif component_type == "tool":
                from ..tool import ToolRegistry

                tool = ToolRegistry.get(component_name)
                if tool:
                    return self._execute_tool(tool, input_data, request)

            # Agents
            elif component_type == "agent":
                # Inbound chat-provider event (ADR-009): route to the ChatBot
                # when the target is one AND the input is the unified inbound
                # event envelope. A chatbot's agent also serves direct calls,
                # so we discriminate by envelope shape, not by identity alone.
                chatbot = self._chatbots.get(component_name) if self._chatbots else None
                if chatbot is not None and self._is_inbound_event(input_data):
                    return self._execute_chat_webhook(chatbot, input_data, request)

                from ..agent import AgentRegistry

                agent = AgentRegistry.get(component_name)
                if agent:
                    return self._execute_agent(agent, input_data, request)

            # Scorers
            elif component_type == "scorer":
                scorer_config = get_builtin_judge_scorer_config(
                    component_name
                ) or ScorerRegistry.get(component_name)
                if scorer_config:
                    return self._execute_scorer(scorer_config, input_data, request)

            # Not found or unsupported
            error_msg = f"Component '{component_name}' of type '{component_type}' not found"
            logger.error(error_msg)

            async def error_response():
                return self._create_error_response(request, error_msg)

            return error_response()

        def handle_message(request: Any) -> Any:
            """Handle incoming execution requests - returns coroutine for Rust to await.

            Wraps the resolved handler coroutine so the running asyncio.Task
            registers itself by run_id, enabling cooperative cancellation
            (a CancelExecution → task.cancel() → CancelledError in the handler).

            resolve_coro runs synchronously here and logs while it dispatches,
            so it gets the run scope too -- otherwise those lines reach the
            control plane unattributed (AGNT5-1070).
            """
            with run_scope(_run_id_of(request.invocation_id)):
                coro = resolve_coro(request)
            return self._track_invocation(request.invocation_id, coro)

        return handle_message

    def _track_invocation(self, invocation_id: str, coro: Any) -> Any:
        """Wrap a handler coroutine so its task self-registers by run_id."""
        run_id = _run_id_of(invocation_id)

        async def _tracked():
            task = asyncio.current_task()
            if task is not None:
                self._inflight[run_id] = task
            try:
                # Binds run_id for every log record emitted while the handler
                # runs, so SDK-internal lines are attributable to the run and
                # not only the ones that happen to print the id (AGNT5-1070).
                with run_scope(run_id):
                    return await coro
            finally:
                self._inflight.pop(run_id, None)

        return _tracked()

    def _on_cancel(self, run_id: str) -> None:
        """Cancel the in-flight invocation for run_id (called from Rust).

        Invoked on a non-loop thread holding the GIL, so schedule the actual
        cancel onto the event loop thread where the task lives.
        """
        loop = self._loop
        if loop is None:
            return

        def _do_cancel() -> None:
            task = self._inflight.get(run_id)
            if task is not None and not task.done():
                task.cancel()

        try:
            loop.call_soon_threadsafe(_do_cancel)
        except RuntimeError:
            # Loop already closed — nothing to cancel.
            pass

    def _is_inbound_event(self, input_data: bytes) -> bool:
        """True if input_data is the unified inbound-event envelope (ADR-009).

        Identified by shape — a `source` plus a string `body` — not a magic
        flag, and distinct from a normal agent input. Lets a chatbot's agent
        still serve direct (non-event) invocations.
        """
        import json

        try:
            data = json.loads(input_data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        return isinstance(data, dict) and "source" in data and "body" in data

    def _execute_chat_webhook(self, chatbot: Any, input_data: bytes, request: Any) -> Any:
        """Execute an inbound chat-provider event via the ChatBot wrapper.

        The gateway delivers the unified envelope (ADR-009 D4):
        { "source": "slack", "event_type": "...", "headers": {...},
          "body": "...", "bot_token": "xoxb-..." }
        The gateway has already verified the signature (D5), so the SDK does
        not re-verify; the bot_token is used to reply.
        """
        import json

        async def _run():
            from .._core import PyExecuteComponentResponse

            try:
                envelope = json.loads(input_data)
                source = envelope["source"]
                headers = envelope.get("headers", {})
                body = envelope.get("body", "").encode("utf-8")
                bot_token = envelope.get("bot_token")

                logger.info(f"Inbound chat event: source={source}, bot={chatbot.name}")

                result = await chatbot.handle_webhook(source, headers, body, bot_token)

                # If the handler returns a challenge response, send it back
                output = json.dumps(result) if result else "{}"
                return PyExecuteComponentResponse(
                    invocation_id=request.invocation_id,
                    success=True,
                    output_data=output.encode("utf-8"),
                    state_update=None,
                    error_message=None,
                    metadata={},
                    event_type="run.completed",
                    content_index=0,
                    sequence=0,
                    attempt=getattr(request, "attempt", 0),
                )
            except Exception as e:
                logger.error(f"Chat event execution failed: {e}", exc_info=True)
                return PyExecuteComponentResponse(
                    invocation_id=request.invocation_id,
                    success=False,
                    output_data=b"",
                    state_update=None,
                    error_message=str(e),
                    metadata={"error_code": type(e).__name__},
                    event_type="run.failed",
                    content_index=0,
                    sequence=0,
                    attempt=getattr(request, "attempt", 0),
                )

        return _run()

    def _print_startup_banner(self, components: list) -> None:
        """Print startup banner with component tree and dashboard link."""
        import os

        visible_components = [comp for comp in components if not _is_system_component(comp)]

        # Group components by type
        by_type: dict[str, list[str]] = {}
        for comp in visible_components:
            comp_type = comp.component_type
            if comp_type not in by_type:
                by_type[comp_type] = []
            by_type[comp_type].append(comp.name)

        # Service header
        print(f"\n  {self.service_name} v{self.service_version}")
        print("  " + "─" * 40)

        # Component tree
        type_order = ["workflow", "function", "agent", "tool", "scorer"]
        type_icons = {
            "workflow": "◆",
            "function": "ƒ",
            "agent": "●",
            "tool": "◇",
            "scorer": "★",
        }

        for comp_type in type_order:
            if comp_type in by_type:
                icon = type_icons.get(comp_type, "•")
                names = by_type[comp_type]
                print(f"  {icon} {comp_type}s ({len(names)})")
                for i, name in enumerate(sorted(names)):
                    is_last = i == len(names) - 1
                    prefix = "└──" if is_last else "├──"
                    print(f"    {prefix} {name}")

        # Dashboard link. `agnt5 dev` injects this only when it can resolve a
        # project-specific Studio URL.
        dashboard_url = os.getenv("AGNT5_DASHBOARD_URL", "").strip()
        print("  " + "─" * 40)
        if dashboard_url:
            print(f"  Dashboard: {dashboard_url}")
        print()

    def _activation_client_for_metadata(
        self, metadata: dict[str, str] | None
    ) -> Any | None:
        if (metadata or {}).get("durable_activation_v1") != "true":
            return None
        if self._activation_client is None:
            from ..exceptions import ActivationError, ActivationErrorCode

            raise ActivationError(
                ActivationErrorCode.DURABILITY_UNAVAILABLE,
                "runtime negotiated durable_activation_v1 but the native activation client is unavailable",
            )
        return self._activation_client

    async def run(self) -> None:
        """Run the worker (register and start message loop).

        This method will:
        1. Discover all registered @function and @workflow handlers
        2. Register with the coordinator
        3. Create a shared Python event loop for all function executions
        4. Enter the message processing loop
        5. Block until shutdown

        This is the main entry point for your worker service.
        """
        try:
            # Register Python handlers for built-in scorers (e.g. llm_judge)
            # that fall through the Rust fast path
            register_builtin_scorer_handlers()

            components = self._discover_components()
            self._print_startup_banner(components)
            self._rust_worker.set_components(components)

            if self.metadata:
                self._rust_worker.set_service_metadata(self.metadata)

            # Configure entity state manager
            if (
                hasattr(self._entity_state_adapter, "_rust_core")
                and self._entity_state_adapter._rust_core
            ):
                self._rust_worker.set_entity_state_manager(self._entity_state_adapter._rust_core)

            loop = asyncio.get_running_loop()
            self._loop = loop
            self._rust_worker.set_event_loop(loop)
            handler = self._create_message_handler()
            self._rust_worker.set_message_handler(handler)
            # Cooperative cancellation: Rust calls this with a run_id when a
            # CancelExecution arrives, and we cancel the matching asyncio.Task.
            self._rust_worker.set_cancel_handler(self._on_cancel)
            self._rust_worker.initialize()

            await self._rust_worker.run()

        except asyncio.CancelledError:
            pass

        except Exception as e:
            logger.error(
                f"Worker failed to start or encountered critical error: {e}",
                exc_info=True,
            )
            _sentry.capture_exception(
                e,
                context={
                    "service_name": self.service_name,
                    "service_version": self.service_version,
                    "error_location": "Worker.run",
                    "error_phase": "worker_lifecycle",
                },
                tags={
                    "sdk_error": "true",
                    "error_type": "worker_failure",
                    "severity": "critical",
                },
                level="error",
            )
            raise

        finally:
            logger.info("Flushing Sentry events before shutdown...")
            _sentry.flush(timeout=5.0)
            logger.info("Worker shutdown complete")
