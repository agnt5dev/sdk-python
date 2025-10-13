# Entity Refactor Plan - Production Readiness

## Current Problems & Proposed Solutions

### 1. **The `__getattribute__` Magic Problem**

**Current Issue:**
```python
def __getattribute__(self, name: str):
    # Lines 348-419: Intercepts EVERY attribute access
    # Wraps methods on every call, checks for '_entity_wrapped', etc.
```

**Why it's bad:**
- Performance hit on EVERY attribute access (even `self._key`)
- Makes debugging nightmare - stack traces become confusing
- IDE can't understand the magic - no autocomplete
- The "mark as wrapped" approach (`_entity_wrapped`) is fragile

**Proposed Solution:**
Wrap methods **once** at class definition time using `__init_subclass__`:

```python
def __init_subclass__(cls):
    # Wrap all async methods ONCE when class is defined
    for name, method in inspect.getmembers(cls):
        if should_wrap_method(name, method):
            wrapped = create_single_writer_wrapper(method)
            setattr(cls, name, wrapped)
```

**Benefits:**
- Zero runtime overhead for attribute access
- Clear stack traces
- IDE understands the methods
- Simpler, more maintainable

---

### 2. **The Testing Complexity Problem**

**Current Issue:**
Tests require complex setup with contextvars:
```python
@pytest.fixture(autouse=True)
def entity_state_manager():
    state_manager = EntityStateManager()
    token = _entity_state_manager_ctx.set(state_manager)
    yield state_manager
    _entity_state_manager_ctx.reset(token)
```

**Why it's bad:**
- Can't test entities without this boilerplate
- Tests coupled to internal implementation
- Hard to mock/stub for unit tests

**Proposed Solutions:**

**Option A: Simple Test Mode**
```python
class Entity:
    @classmethod
    def create_for_testing(cls, key: str):
        """Create entity with isolated test state."""
        instance = cls(key)
        instance._test_mode = True
        instance._test_state = {}
        return instance
```

**Option B: Keep contextvars but simplify**
```python
# Provide a simple helper
from agnt5.entity import with_test_context

@with_test_context
async def test_my_entity():
    cart = ShoppingCart(key="test")
    await cart.add_item("item", 1, 10.0)
```

**I lean toward Option A** - it's simpler and doesn't leak implementation details.

---

### 3. **The WorkflowEntity Problem**

**Current Issue:**
Lines 464-613 add WorkflowEntity with complex tracking:
```python
class WorkflowEntity(Entity):
    def __init__(self, run_id: str):
        self._step_events = []
        self._completed_steps = {}
        self._state_changes = []
```

**Why it's bad:**
- Wrong module - should be in workflow.py
- Mixes concerns (entities shouldn't know about workflows)
- Adds complexity without clear value
- The state tracking duplicates what Context already does

**Proposed Solution:**
Move it to workflow.py or remove entirely. Workflows can use regular entities if needed:
```python
# In workflow.py
class WorkflowState(Entity):
    """Entity specifically for workflow state if needed."""
    pass
```

---

### 4. **Missing Error Context**

**Current Issue:**
```python
raise RuntimeError(
    "Entity state manager not set in context. "
    "Entities must be executed within a Worker context."
)
```

**Why it's bad:**
- Doesn't tell developer HOW to fix it
- No guidance for testing scenarios

**Proposed Solution:**
```python
raise RuntimeError(
    "Entity execution requires a state manager context.\n"
    "\n"
    "If you're in production:\n"
    "  Entities must be executed through Worker.execute()\n"
    "\n"
    "If you're testing:\n"
    "  cart = ShoppingCart.create_for_testing('key')\n"
    "  # or\n"
    "  with entity_test_context():\n"
    "      cart = ShoppingCart('key')\n"
)
```

---

### 5. **State Access Inconsistency**

**Current Issue:**
```python
# In entity methods
self.state.get("key")  # Simple, clean

# In tests (line 279 of test_entity.py)
entity_state_manager.get_state("Counter", "test")  # Different API
```

**Proposed Solution:**
Keep entity methods clean, but provide test helpers:
```python
# In tests
cart = ShoppingCart.create_for_testing("test")
await cart.add_item("item", 1, 10.0)
assert cart.state.get("items") == {"item": {"quantity": 1, "price": 10.0}}
# State access is the same!
```

---

### 6. **The Instance Cache Complexity**

**Current Issue:**
```python
_ENTITY_INSTANCE_CACHE: WeakValueDictionary = WeakValueDictionary()
```

**Question:** Do we really need this? What problem does it solve?

**Concerns:**
- Adds complexity
- Could cause surprising behavior (same instance returned)
- Memory management becomes tricky

**Recommendation:** Remove it unless there's a strong use case.

---

## Simplified Architecture Proposal

### Core Principle: Keep It Simple

```python
class Entity:
    """Simple, clear entity with explicit behavior."""

    def __init__(self, key: str):
        self.key = key
        self.state = None  # Set up by wrapper
        self._entity_type = self.__class__.__name__

    def __init_subclass__(cls):
        """Wrap methods at class definition time."""
        for name, method in get_async_methods(cls):
            wrapped = wrap_with_single_writer(method, cls.__name__)
            setattr(cls, name, wrapped)

        # Register for platform discovery
        EntityRegistry.register(EntityType(cls.__name__, cls))
```

### Method Wrapping (Simple & Clear)

```python
def wrap_with_single_writer(method, entity_type):
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        # Get state manager (with good error if missing)
        manager = get_state_manager_with_helpful_errors()

        # Get lock for single-writer
        lock = manager.get_lock((entity_type, self.key))

        async with lock:
            # Set up state
            state_dict = manager.get_state((entity_type, self.key))
            self.state = EntityState(state_dict)

            # Execute
            try:
                return await method(self, *args, **kwargs)
            finally:
                # Clean up if needed
                pass

    return wrapper
```

### Testing Support (Developer Friendly)

```python
# Option 1: Class method
cart = ShoppingCart.create_for_testing("test-key")
await cart.add_item("item", 1, 10.0)

# Option 2: Context manager
with entity_test_context():
    cart = ShoppingCart("test-key")
    await cart.add_item("item", 1, 10.0)

# Option 3: Decorator
@with_entity_test_context
async def test_cart():
    cart = ShoppingCart("test-key")
    await cart.add_item("item", 1, 10.0)
```

---

## What About Performance?

### Current Performance Issues:
1. `__getattribute__` called on EVERY attribute access
2. Method wrapping happens on every call
3. Checks like `hasattr(attr, '_entity_wrapped')` on every method call

### Proposed Performance Wins:
1. Methods wrapped ONCE at class definition
2. No runtime `__getattribute__` overhead
3. Direct method calls after initial setup

### Benchmarking Needed:
- Measure current vs proposed for 1000 method calls
- Memory usage with/without instance cache
- Lock contention under load

---

## Migration Path

### Step 1: Core Refactor (2-3 hours)
- Remove `__getattribute__`
- Implement `__init_subclass__` wrapping
- Simplify error messages

### Step 2: Move WorkflowEntity (30 min)
- Move to workflow.py
- Or remove if not needed

### Step 3: Testing Support (1-2 hours)
- Add `create_for_testing()` method
- Update test fixtures
- Document testing patterns

### Step 4: Update Tests (2-3 hours)
- Update all tests to use new patterns
- Ensure 100% test coverage
- Add performance tests

### Step 5: Documentation (1-2 hours)
- Update examples
- Write migration guide
- Document best practices

---

## Questions to Answer First

1. **Do we need the instance cache?**
   - What problem does it solve?
   - Is the complexity worth it?

2. **Do we need WorkflowEntity at all?**
   - Can workflows just use regular entities?
   - What specific workflow needs does it address?

3. **Testing approach preference?**
   - Simple test mode on entity?
   - Keep contextvars but simplify?
   - Both options?

4. **Should entities be allowed to call other entities?**
   - Currently nothing prevents it
   - Could lead to deadlocks with locks

5. **Phase 2 persistence - any constraints?**
   - Will state be loaded lazily or eagerly?
   - How will versioning work?
   - Should we prepare for this now?

---

## Recommendation

**Go with the simplest approach that works:**

1. **Remove all magic** (`__getattribute__`)
2. **Wrap at class definition time** (clean, fast)
3. **Simple test mode** (no contextvars in tests)
4. **Move/remove WorkflowEntity**
5. **Clear error messages**

This gets us to 8/10 production ready with:
- ✅ Clean, debuggable code
- ✅ Good performance
- ✅ Easy testing
- ✅ Clear error messages
- ✅ No surprising behavior

To get to 9/10, we'd add:
- Distributed lock support (ready for multi-worker)
- State persistence hooks (ready for Phase 2)
- Performance benchmarks
- Comprehensive examples

---

## Alternative: Even Simpler?

We could go even simpler with explicit decorators:

```python
class ShoppingCart(Entity):
    @entity_method  # Explicit is better than implicit
    async def add_item(self, item_id: str, quantity: int):
        items = self.state.get("items", {})
        items[item_id] = quantity
        self.state.set("items", items)

    # This method wouldn't have single-writer guarantee
    async def some_helper(self):
        return "not wrapped"
```

**Pros:** Very explicit, no magic
**Cons:** Developer must remember to add decorator

What do you think?