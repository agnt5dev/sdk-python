# Worker-Scoped Registration Plan

**Status:** Planned (Not Implemented)
**Related:** Step 8 of SDK Refactoring Roadmap (see `function-refactor.md`)

---

## Problem Statement

### Current State

All component registries in the AGNT5 SDK use **global module-level dictionaries**:

```python
# sdk/sdk-python/src/agnt5/function.py
_FUNCTION_REGISTRY: Dict[str, FunctionConfig] = {}  # Global!

# sdk/sdk-python/src/agnt5/workflow.py
_WORKFLOW_REGISTRY: Dict[str, WorkflowConfig] = {}  # Global!

# sdk/sdk-python/src/agnt5/entity.py
_ENTITY_REGISTRY: Dict[str, EntityType] = {}  # Global!
```

Decorators register components **at import time** (when modules are loaded):

```python
@function  # Registers to global _FUNCTION_REGISTRY immediately on import
async def greet(ctx: Context, name: str) -> str:
    return f"Hello, {name}!"
```

### Issues with Global Registries

1. **No Isolation** - All Workers in the same process share the same global registry
   - Can't run multiple workers with different component sets
   - Services must be in separate processes

2. **Testing Friction** - Tests share global state
   - Must manually call `FunctionRegistry.clear()` between tests
   - Can't run tests in parallel (they interfere with each other)
   - Difficult to test component registration logic

3. **No Dynamic Loading** - Components registered at import time
   - Can't load/unload components per worker at runtime
   - Can't have different component versions in same process

4. **Name Collisions** - Global namespace issues
   - Two modules can't define functions with same name
   - Importing both causes registration errors

5. **Tight Coupling** - Worker depends on global state
   - Hard to mock or test worker discovery
   - Can't control which components a worker sees

### Example Problem: Multi-Worker Application

```python
# This DOESN'T WORK with current global registry approach:

# service_a.py
@function
async def process_data(ctx, data):  # Registers globally
    return data.upper()

# service_b.py
@function
async def process_data(ctx, data):  # ERROR: Name collision!
    return data.lower()

# main.py
import service_a
import service_b

# Can't run both workers in same process:
worker_a = Worker(service_name="service-a")  # Wants service_a.process_data
worker_b = Worker(service_name="service-b")  # Wants service_b.process_data
# Both see the same global registry - conflict!
```

---

## Proposed Solution: Worker-Scoped Registries

Each `Worker` instance maintains its own **isolated component registry**, discovered at Worker initialization instead of module import time.

### Key Design Principles

1. **Backward Compatible** - Existing code continues to work (Phase 1)
2. **Gradual Migration** - Opt-in scoped registries, then deprecate global
3. **Flexible Discovery** - Support module-based and explicit registration
4. **Clear Ownership** - Workers own their registries, not global state

---

## Implementation Plan

### Phase 1: Opt-In Worker-Scoped Registries (Backward Compatible)

**Goal:** Add worker-scoped registration as an opt-in feature while maintaining global registry as default.

#### 1.1 Create Worker-Scoped Registry Classes

**File:** `sdk/sdk-python/src/agnt5/registry.py` (new file)

```python
"""Component registry implementation for AGNT5 SDK."""

from typing import Dict, Optional
from .types import FunctionConfig, WorkflowConfig
from .entity import EntityType
from .tool import Tool
from .agent import Agent


class ComponentRegistry:
    """Worker-scoped component registry.

    Provides isolated storage for functions, workflows, entities, tools, and agents.
    Each Worker instance can maintain its own registry for component isolation.

    Example:
        ```python
        registry = ComponentRegistry()
        registry.register_function(config)
        config = registry.get_function("my_function")
        ```
    """

    def __init__(self):
        """Initialize empty registry."""
        self.functions: Dict[str, FunctionConfig] = {}
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.entities: Dict[str, EntityType] = {}
        self.tools: Dict[str, Tool] = {}
        self.agents: Dict[str, Agent] = {}

    # Function registry methods
    def register_function(self, config: FunctionConfig) -> None:
        """Register a function in this registry."""
        if config.name in self.functions:
            raise ValueError(f"Function '{config.name}' already registered")
        self.functions[config.name] = config

    def get_function(self, name: str) -> Optional[FunctionConfig]:
        """Get function by name."""
        return self.functions.get(name)

    def all_functions(self) -> Dict[str, FunctionConfig]:
        """Get all registered functions."""
        return self.functions.copy()

    # Similar methods for workflows, entities, tools, agents...

    def clear(self) -> None:
        """Clear all registered components."""
        self.functions.clear()
        self.workflows.clear()
        self.entities.clear()
        self.tools.clear()
        self.agents.clear()
```

#### 1.2 Update Worker Class

**File:** `sdk/sdk-python/src/agnt5/worker.py`

**Changes:**

1. Add `use_scoped_registry` parameter to `Worker.__init__()`
2. Create worker-scoped registry when enabled
3. Add explicit registration methods
4. Update discovery and message handling to use scoped registry

**Implementation:**

```python
from .registry import ComponentRegistry

class Worker:
    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        coordinator_endpoint: Optional[str] = None,
        runtime: str = "standalone",
        metadata: Optional[Dict[str, str]] = None,
        use_scoped_registry: bool = False,  # NEW: Enable scoped registry
    ):
        """Initialize a new Worker.

        Args:
            service_name: Unique name for this service
            service_version: Version string
            coordinator_endpoint: Coordinator endpoint URL
            runtime: Runtime type
            metadata: Optional service-level metadata
            use_scoped_registry: If True, use worker-scoped registry instead of global
        """
        self.service_name = service_name
        self.service_version = service_version
        self.coordinator_endpoint = coordinator_endpoint
        self.runtime = runtime
        self.metadata = metadata or {}

        # NEW: Create scoped registry if enabled
        self.use_scoped_registry = use_scoped_registry
        if use_scoped_registry:
            self._registry = ComponentRegistry()
            logger.info("Worker using scoped component registry")
        else:
            self._registry = None
            logger.debug("Worker using global component registry")

        # ... rest of __init__

    # NEW: Explicit registration methods
    def register_function(self, func: Callable) -> None:
        """Register a function with this worker's scoped registry.

        Args:
            func: Function decorated with @function

        Raises:
            ValueError: If worker not configured for scoped registry

        Example:
            ```python
            @function
            async def my_func(ctx: Context, data: str) -> str:
                return data.upper()

            worker = Worker(service_name="test", use_scoped_registry=True)
            worker.register_function(my_func)
            ```
        """
        if not self.use_scoped_registry:
            raise ValueError(
                "Worker not configured for scoped registry. "
                "Set use_scoped_registry=True when creating Worker."
            )

        # Extract config from decorated function
        if not hasattr(func, '_agnt5_config'):
            raise ValueError(f"Function {func.__name__} is not decorated with @function")

        config = func._agnt5_config
        self._registry.register_function(config)
        logger.debug(f"Registered function '{config.name}' with worker")

    def register_workflow(self, workflow: Callable) -> None:
        """Register a workflow with this worker's scoped registry."""
        if not self.use_scoped_registry:
            raise ValueError("Worker not configured for scoped registry")

        if not hasattr(workflow, '_agnt5_config'):
            raise ValueError(f"Workflow {workflow.__name__} is not decorated with @workflow")

        config = workflow._agnt5_config
        self._registry.workflows[config.name] = config
        logger.debug(f"Registered workflow '{config.name}' with worker")
```

#### 1.3 Update Discovery Logic

**File:** `sdk/sdk-python/src/agnt5/worker.py`

**Method:** `_discover_components()`

```python
def _discover_components(self):
    """Discover all registered components.

    Uses either scoped registry (if enabled) or global registries (default).
    """
    components = []

    if self.use_scoped_registry:
        # Discover from worker's scoped registry
        logger.debug("Discovering components from worker-scoped registry")

        # Discover functions from scoped registry
        for name, config in self._registry.all_functions().items():
            input_schema_str = json.dumps(config.input_schema) if config.input_schema else None
            output_schema_str = json.dumps(config.output_schema) if config.output_schema else None

            component_info = self._PyComponentInfo(
                name=name,
                component_type="function",
                metadata=config.metadata or {},
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered function: {name}")

        # Similar for workflows, entities, tools, agents from self._registry
        # ...

    else:
        # Existing behavior - discover from global registries
        logger.debug("Discovering components from global registries")

        from .tool import ToolRegistry
        from .entity import EntityRegistry
        from .agent import AgentRegistry

        # Discover functions from global registry
        for name, config in FunctionRegistry.all().items():
            # ... existing code

        # ... rest of existing discovery logic

    logger.info(f"Discovered {len(components)} components")
    return components
```

#### 1.4 Update Message Handler

**File:** `sdk/sdk-python/src/agnt5/worker.py`

**Method:** `_create_message_handler()`

```python
def _create_message_handler(self):
    """Create the message handler that will be called by Rust worker."""

    def handle_message(request):
        """Handle incoming execution requests."""
        component_name = request.component_name
        component_type = request.component_type
        input_data = request.input_data

        logger.debug(
            f"Handling {component_type} request: {component_name}"
        )

        # Route based on component type
        if component_type == "function":
            if self.use_scoped_registry:
                # Look up in worker's scoped registry
                config = self._registry.get_function(component_name)
            else:
                # Look up in global registry
                config = FunctionRegistry.get(component_name)

            if config:
                logger.debug(f"Found function: {component_name}")
                return self._execute_function(config, input_data, request)

        elif component_type == "workflow":
            if self.use_scoped_registry:
                config = self._registry.workflows.get(component_name)
            else:
                config = WorkflowRegistry.get(component_name)

            if config:
                logger.debug(f"Found workflow: {component_name}")
                return self._execute_workflow(config, input_data, request)

        # ... similar for entities, tools, agents

        # Not found
        error_msg = f"Component '{component_name}' of type '{component_type}' not found"
        logger.error(error_msg)

        async def error_response():
            return self._create_error_response(request, error_msg)

        return error_response()

    return handle_message
```

#### 1.5 Usage Example (Phase 1)

```python
from agnt5 import Worker, function, Context

# Define functions (still uses decorator)
@function
async def greet(ctx: Context, name: str) -> str:
    return f"Hello, {name}!"

@function
async def farewell(ctx: Context, name: str) -> str:
    return f"Goodbye, {name}!"

# Option A: Global registry (existing behavior - default)
worker_global = Worker(service_name="greeter")
# Automatically discovers all @function decorated functions

# Option B: Scoped registry (new - opt-in)
worker_scoped = Worker(
    service_name="greeter-scoped",
    use_scoped_registry=True  # Enable scoped registry
)
# Must explicitly register functions
worker_scoped.register_function(greet)
worker_scoped.register_function(farewell)

await worker_scoped.run()
```

---

### Phase 2: Module-Based Auto-Discovery

**Goal:** Auto-discover components from specified modules without requiring explicit registration.

#### 2.1 Add Module Discovery to Worker

**File:** `sdk/sdk-python/src/agnt5/worker.py`

```python
from typing import List, Optional
from types import ModuleType
import inspect

class Worker:
    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        coordinator_endpoint: Optional[str] = None,
        runtime: str = "standalone",
        metadata: Optional[Dict[str, str]] = None,
        use_scoped_registry: bool = False,
        modules: Optional[List[ModuleType]] = None,  # NEW: Auto-discover from modules
    ):
        """Initialize a new Worker.

        Args:
            modules: List of modules to scan for components. If provided,
                     automatically enables use_scoped_registry=True.

        Example:
            ```python
            import my_functions
            import my_workflows

            worker = Worker(
                service_name="my-service",
                modules=[my_functions, my_workflows]  # Auto-discover
            )
            ```
        """
        self.service_name = service_name
        self.service_version = service_version
        self.coordinator_endpoint = coordinator_endpoint
        self.runtime = runtime
        self.metadata = metadata or {}

        # If modules provided, force scoped registry
        if modules:
            use_scoped_registry = True
            logger.info(f"Module-based discovery enabled for {len(modules)} modules")

        self.use_scoped_registry = use_scoped_registry
        if use_scoped_registry:
            self._registry = ComponentRegistry()
        else:
            self._registry = None

        # NEW: Discover from modules if provided
        if modules:
            self._discover_from_modules(modules)

        # ... rest of __init__

    def _discover_from_modules(self, modules: List[ModuleType]) -> None:
        """Scan modules for decorated components and register them.

        Looks for functions/workflows/entities decorated with @function, @workflow, etc.
        and registers them with the worker's scoped registry.

        Args:
            modules: List of modules to scan
        """
        logger.info(f"Scanning {len(modules)} modules for components...")

        discovered = {
            'functions': 0,
            'workflows': 0,
            'entities': 0,
            'tools': 0,
            'agents': 0,
        }

        for module in modules:
            logger.debug(f"Scanning module: {module.__name__}")

            for name, obj in inspect.getmembers(module):
                # Skip private members
                if name.startswith('_'):
                    continue

                # Check if it has _agnt5_config attribute (decorated component)
                if hasattr(obj, '_agnt5_config'):
                    config = obj._agnt5_config

                    # Register based on config type
                    from .types import FunctionConfig, WorkflowConfig

                    if isinstance(config, FunctionConfig):
                        self._registry.register_function(config)
                        discovered['functions'] += 1
                        logger.debug(f"  Found function: {config.name}")

                    elif isinstance(config, WorkflowConfig):
                        self._registry.workflows[config.name] = config
                        discovered['workflows'] += 1
                        logger.debug(f"  Found workflow: {config.name}")

                    # TODO: Add similar checks for entities, tools, agents

        logger.info(
            f"Module discovery complete: "
            f"{discovered['functions']} functions, "
            f"{discovered['workflows']} workflows, "
            f"{discovered['entities']} entities, "
            f"{discovered['tools']} tools, "
            f"{discovered['agents']} agents"
        )
```

#### 2.2 Usage Example (Phase 2)

```python
# my_functions.py
from agnt5 import function, Context

@function
async def greet(ctx: Context, name: str) -> str:
    return f"Hello, {name}!"

@function
async def farewell(ctx: Context, name: str) -> str:
    return f"Goodbye, {name}!"

# main.py
from agnt5 import Worker
import my_functions

# Auto-discover all functions from my_functions module
worker = Worker(
    service_name="greeter",
    modules=[my_functions]  # No need for explicit registration!
)

await worker.run()
```

---

### Phase 3: Deprecate Global Registries (Future Breaking Change)

**Goal:** Eventually remove global registries entirely.

#### 3.1 Add Deprecation Warnings

**File:** `sdk/sdk-python/src/agnt5/function.py`

```python
import warnings

class FunctionRegistry:
    @staticmethod
    def register(config: FunctionConfig) -> None:
        """Register a function handler."""
        # Emit deprecation warning
        warnings.warn(
            "Global function registry is deprecated and will be removed in v2.0. "
            "Use Worker(modules=[...]) or Worker.register_function() instead. "
            "See migration guide: https://docs.agnt5.dev/migration/scoped-registries",
            DeprecationWarning,
            stacklevel=3
        )

        # ... existing registration logic
```

#### 3.2 Migration Guide

**Document:** Update SDK docs with migration guide

**Before (Global Registry - Deprecated):**
```python
from agnt5 import Worker, function, Context

@function
async def greet(ctx: Context, name: str) -> str:
    return f"Hello, {name}!"

worker = Worker(service_name="greeter")  # Uses global registry
await worker.run()
```

**After (Scoped Registry - Recommended):**

**Option 1: Module-based discovery (Recommended)**
```python
from agnt5 import Worker
import my_functions  # Contains @function decorated functions

worker = Worker(
    service_name="greeter",
    modules=[my_functions]  # Auto-discover from module
)
await worker.run()
```

**Option 2: Explicit registration**
```python
from agnt5 import Worker, function, Context

@function
async def greet(ctx: Context, name: str) -> str:
    return f"Hello, {name}!"

worker = Worker(service_name="greeter", use_scoped_registry=True)
worker.register_function(greet)
await worker.run()
```

**Option 3: Package-level discovery**
```python
from agnt5 import Worker
import my_service  # Package with __init__.py that imports all components

worker = Worker(
    service_name="my-service",
    modules=[my_service]
)
await worker.run()
```

---

## Testing Improvements

### Before: Global Registry (Tests Interfere)

```python
# test_service_a.py
from agnt5 import function, FunctionRegistry

@function
async def process(ctx, data):
    return data.upper()

def test_process():
    config = FunctionRegistry.get("process")
    assert config is not None

# Problem: If test_service_b.py also defines 'process', tests can't run in parallel!
```

### After: Scoped Registry (Isolated Tests)

```python
# test_service_a.py
from agnt5 import Worker, function

@function
async def process(ctx, data):
    return data.upper()

def test_process():
    worker = Worker(service_name="test", use_scoped_registry=True)
    worker.register_function(process)

    config = worker._registry.get_function("process")
    assert config is not None
    # Tests are isolated - no global state!

# Can run in parallel with other tests - no interference!
```

### Pytest Fixtures

```python
import pytest
from agnt5 import Worker

@pytest.fixture
def worker():
    """Create isolated worker for each test."""
    return Worker(
        service_name="test",
        use_scoped_registry=True
    )

def test_function_a(worker):
    @function
    async def my_func(ctx, x):
        return x * 2

    worker.register_function(my_func)
    # Test in complete isolation

def test_function_b(worker):
    # Different worker instance - no interference!
    @function
    async def my_func(ctx, x):  # Same name, different function
        return x + 2

    worker.register_function(my_func)
```

---

## Implementation Checklist

### Phase 1: Opt-In Scoped Registries (Backward Compatible)

- [ ] **Create `registry.py`**
  - [ ] Implement `ComponentRegistry` class
  - [ ] Add methods for all component types (functions, workflows, entities, tools, agents)
  - [ ] Add clear() and introspection methods

- [ ] **Update `worker.py`**
  - [ ] Add `use_scoped_registry` parameter to `Worker.__init__()`
  - [ ] Create `self._registry` when scoped registry enabled
  - [ ] Add `Worker.register_function()` method
  - [ ] Add `Worker.register_workflow()` method
  - [ ] Add similar methods for entities, tools, agents
  - [ ] Update `_discover_components()` to support scoped registry
  - [ ] Update `_create_message_handler()` to support scoped registry

- [ ] **Testing**
  - [ ] Unit tests for `ComponentRegistry` class
  - [ ] Tests for scoped vs global registry behavior
  - [ ] Tests for explicit registration methods
  - [ ] Parallel test execution tests

- [ ] **Documentation**
  - [ ] Document `use_scoped_registry` parameter
  - [ ] Add examples of explicit registration
  - [ ] Update API reference

### Phase 2: Module-Based Discovery

- [ ] **Update `worker.py`**
  - [ ] Add `modules` parameter to `Worker.__init__()`
  - [ ] Implement `_discover_from_modules()` method
  - [ ] Auto-enable scoped registry when modules provided
  - [ ] Support package-level discovery

- [ ] **Testing**
  - [ ] Tests for module-based discovery
  - [ ] Tests for multiple modules
  - [ ] Tests for package discovery
  - [ ] Tests for missing decorators

- [ ] **Documentation**
  - [ ] Document `modules` parameter
  - [ ] Add module-based discovery examples
  - [ ] Update best practices guide

### Phase 3: Deprecation (Future Breaking Change)

- [ ] **Add Deprecation Warnings**
  - [ ] Warning in `FunctionRegistry.register()`
  - [ ] Warning in `WorkflowRegistry.register()`
  - [ ] Warning in `EntityRegistry.register()`
  - [ ] Warning in other global registries

- [ ] **Create Migration Guide**
  - [ ] Before/after examples
  - [ ] Migration strategies (module-based, explicit, package-level)
  - [ ] Breaking changes documentation
  - [ ] Timeline for removal

- [ ] **Update All Examples**
  - [ ] Convert to module-based discovery
  - [ ] Update README examples
  - [ ] Update tutorial code

- [ ] **Remove Global Registries** (v2.0)
  - [ ] Remove `_FUNCTION_REGISTRY` global
  - [ ] Remove `_WORKFLOW_REGISTRY` global
  - [ ] Remove `_ENTITY_REGISTRY` global
  - [ ] Remove global registry classes
  - [ ] Make scoped registry mandatory

---

## Files to Modify

### New Files
- `sdk/sdk-python/src/agnt5/registry.py` - ComponentRegistry class

### Modified Files
- `sdk/sdk-python/src/agnt5/worker.py` - Add scoped registry support
- `sdk/sdk-python/src/agnt5/function.py` - Add deprecation warnings (Phase 3)
- `sdk/sdk-python/src/agnt5/workflow.py` - Add deprecation warnings (Phase 3)
- `sdk/sdk-python/src/agnt5/entity.py` - Add deprecation warnings (Phase 3)
- `sdk/sdk-python/src/agnt5/tool.py` - Add deprecation warnings (Phase 3)
- `sdk/sdk-python/src/agnt5/agent.py` - Add deprecation warnings (Phase 3)

### Documentation Files
- `sdk/sdk-python/README.md` - Update with scoped registry examples
- New migration guide documentation

---

## Benefits

### Isolation
- Multiple workers in same process with different components
- No global state contamination
- Clear component ownership per worker

### Testing
- Parallel test execution without interference
- No manual registry cleanup between tests
- Easy fixture-based test isolation
- Test different component configurations easily

### Flexibility
- Dynamic component loading/unloading
- Runtime component registration
- Multiple versions of same component in different workers
- Choose between simple (global) or isolated (scoped) approach

### Maintainability
- Clear separation of concerns
- Worker owns its components
- Easier to reason about component lifecycle
- Better support for multi-service applications

### Migration Path
- Gradual adoption - no breaking changes initially
- Multiple migration strategies (module-based, explicit, package-level)
- Backward compatible until v2.0
- Clear deprecation timeline

---

## Estimated Effort

### Development Time
- **Phase 1 (Opt-in scoped registries):** 2-3 days
  - ComponentRegistry implementation: 0.5 days
  - Worker updates: 1 day
  - Testing: 0.5-1 day
  - Documentation: 0.5 day

- **Phase 2 (Module discovery):** 1-2 days
  - Module scanning implementation: 0.5 day
  - Testing: 0.5 day
  - Documentation and examples: 0.5 day

- **Phase 3 (Deprecation - future):** 1 day
  - Deprecation warnings: 0.25 day
  - Migration guide: 0.5 day
  - Update examples: 0.25 day

**Total Estimated Effort:** 4-6 days for complete implementation across all phases

### Testing Effort
- Unit tests: 1 day
- Integration tests: 0.5 day
- Migration testing: 0.5 day

**Total with Testing:** 6-8 days

---

## Open Questions

1. **Context Manager API?** Should we add a context manager API for scoped registration?
   ```python
   with worker.component_scope():
       @function
       def my_func(...):
           pass
   ```

2. **Auto-discovery heuristics?** Should we support automatic module discovery (scan `sys.modules`)?

3. **Registry serialization?** Should ComponentRegistry support save/load for caching?

4. **Hot reload?** Should we support dynamic component reloading during worker runtime?

5. **Nested registries?** Should we support registry inheritance/composition?

---

## Related Issues

- See `function-refactor.md` for broader SDK refactoring roadmap
- Step 8 of 10-step refactoring plan
- Complements other improvements (type-safety, Pydantic integration, optional context)

---

## References

- **Cloudflare Durable Objects:** Similar class-based component registration
- **Flask Application Factories:** Pattern for creating app instances with isolated state
- **FastAPI Dependency Injection:** Request-scoped dependencies vs global singletons
- **pytest fixtures:** Isolated test state management
