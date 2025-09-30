"""
Component abstraction layer for AGNT5 SDK.

This module defines the base classes for all component types:
- Functions: Stateless operations
- Objects: Virtual objects with persistent state  
- Flows: Multi-step workflows with orchestration
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
import uuid
import json
import time


class ComponentType(Enum):
    """Component types matching protobuf enum"""
    FUNCTION = "function"
    OBJECT = "object" 
    FLOW = "flow"


class ExecutionContext:
    """
    Unified execution context for all component types.
    
    Provides methods for:
    - Functions: Simple input/output
    - Objects: State management and mutations
    - Flows: Orchestration and step coordination
    """
    
    def __init__(self, invocation_id: str, component_type: ComponentType):
        self.invocation_id = invocation_id
        self.component_type = component_type
        
        # Object-specific state management
        self.object_id: Optional[str] = None
        self.state: Optional[Dict[str, Any]] = None
        self.state_mutations: List[Dict[str, Any]] = []
        
        # Flow-specific orchestration
        self.flow_instance_id: Optional[str] = None
        self.flow_step: int = 0
        self.checkpoint_data: Optional[Dict[str, Any]] = None
        
        # Extensible metadata
        self.metadata: Dict[str, str] = {}
    
    # State management methods (for Objects)
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from object state"""
        if self.state is None:
            return default
        return self.state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a value in object state (records mutation)"""
        if self.state is None:
            self.state = {}
        
        old_value = self.state.get(key)
        self.state[key] = value
        
        # Record mutation for persistence
        self.state_mutations.append({
            "operation": "set",
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "timestamp": int(time.time() * 1000)
        })
    
    def delete_state(self, key: str) -> Any:
        """Delete a value from object state"""
        if self.state is None or key not in self.state:
            return None
            
        old_value = self.state.pop(key)
        
        # Record mutation
        self.state_mutations.append({
            "operation": "delete", 
            "key": key,
            "old_value": old_value,
            "new_value": None,
            "timestamp": int(time.time() * 1000)
        })
        
        return old_value
    
    # Flow orchestration methods (for Flows - future implementation)
    async def call_function(self, function_name: str, input_data: Any) -> Any:
        """Call another function from within a flow"""
        # TODO: Implement in Phase 3 (Flows)
        raise NotImplementedError("Flow orchestration coming in Phase 3")
    
    async def call_object(self, object_type: str, object_id: str, 
                         method: str, input_data: Any) -> Any:
        """Call a method on a virtual object from within a flow"""
        # TODO: Implement in Phase 3 (Flows)
        raise NotImplementedError("Flow orchestration coming in Phase 3")
    
    async def sleep(self, duration_seconds: int) -> None:
        """Durable sleep in a flow"""
        # TODO: Implement in Phase 3 (Flows)
        raise NotImplementedError("Flow orchestration coming in Phase 3")
    
    async def wait_for_event(self, event_type: str, timeout_seconds: int = None) -> Any:
        """Wait for external event in a flow"""
        # TODO: Implement in Phase 3 (Flows)
        raise NotImplementedError("Flow orchestration coming in Phase 3")


class Component(ABC):
    """Base class for all component types"""
    
    def __init__(self, name: str, component_type: ComponentType):
        self.name = name
        self.component_type = component_type
        self.metadata: Dict[str, str] = {}
    
    @abstractmethod
    async def invoke(self, context: ExecutionContext, input_data: Any) -> Any:
        """Execute the component with given context and input"""
        pass
    
    def to_component_info(self) -> Dict[str, Any]:
        """Convert to ComponentInfo for registration"""
        return {
            "name": self.name,
            "component_type": self.component_type.value,
            "metadata": self.metadata
        }


class FunctionComponent(Component):
    """Function component - stateless operation"""
    
    def __init__(self, name: str, handler: Callable, **kwargs):
        super().__init__(name, ComponentType.FUNCTION)
        self.handler = handler
        self.streaming = kwargs.get('streaming', False)
        
        # Add function-specific metadata
        self.metadata.update({
            'streaming': str(self.streaming),
            'handler_name': handler.__name__
        })
    
    async def invoke(self, context: ExecutionContext, input_data: Any) -> Any:
        """Execute the function"""
        # Functions get simple context and input
        if self.streaming:
            # For streaming functions, return async generator
            result = self.handler(context, input_data)
            if hasattr(result, '__aiter__'):
                return result
            else:
                # Convert sync generator to async
                async def async_generator():
                    for item in result:
                        yield item
                return async_generator()
        else:
            # Regular function call
            result = self.handler(context, input_data)
            # Handle both sync and async functions
            if hasattr(result, '__await__'):
                return await result
            return result


class ObjectComponent(Component):
    """Virtual Object component - stateful entity"""
    
    def __init__(self, name: str, object_class: Type, **kwargs):
        super().__init__(name, ComponentType.OBJECT)
        self.object_class = object_class
        
        # Add object-specific metadata
        self.metadata.update({
            'class_name': object_class.__name__,
            'methods': [m for m in dir(object_class) 
                       if not m.startswith('_') and callable(getattr(object_class, m))]
        })
    
    async def invoke(self, context: ExecutionContext, input_data: Any) -> Any:
        """Execute a method on the virtual object"""
        # TODO: Implement in Phase 2 (Objects)
        # For now, raise helpful error
        raise NotImplementedError(
            f"Virtual Objects coming in Phase 2. "
            f"Component '{self.name}' is registered but not yet executable. "
            f"Use @function decorator for now."
        )


class FlowComponent(Component):
    """Flow component - multi-step workflow"""
    
    def __init__(self, name: str, flow_handler: Callable, **kwargs):
        super().__init__(name, ComponentType.FLOW)
        self.flow_handler = flow_handler
        self.steps = kwargs.get('steps', [])
        
        # Add flow-specific metadata
        self.metadata.update({
            'handler_name': flow_handler.__name__,
            'step_count': str(len(self.steps)) if self.steps else 'dynamic'
        })
    
    async def invoke(self, context: ExecutionContext, input_data: Any) -> Any:
        """Execute the workflow"""
        # TODO: Implement in Phase 3 (Flows)
        # For now, raise helpful error
        raise NotImplementedError(
            f"Flows/Workflows coming in Phase 3. "
            f"Component '{self.name}' is registered but not yet executable. "
            f"Use @function decorator for now."
        )


# Helper classes for future phases

class StateManager:
    """Manages state persistence for virtual objects (Phase 2)"""
    
    def __init__(self):
        # Will be implemented with actual state backend
        pass
    
    async def load_state(self, object_type: str, object_id: str) -> Optional[Dict[str, Any]]:
        """Load object state from persistent storage"""
        # TODO: Implement with NATS KV or similar
        return None
    
    async def save_state(self, object_type: str, object_id: str, 
                        state: Dict[str, Any], 
                        mutations: List[Dict[str, Any]]) -> None:
        """Save object state to persistent storage"""
        # TODO: Implement with NATS KV or similar
        pass


class FlowExecutor:
    """Manages workflow execution and orchestration (Phase 3)"""
    
    def __init__(self):
        # Will be implemented with actual flow execution engine
        pass
    
    async def execute_step(self, flow_instance_id: str, step: int, 
                          input_data: Any) -> Any:
        """Execute a single step in a workflow"""
        # TODO: Implement with deterministic replay
        pass
    
    async def checkpoint(self, flow_instance_id: str, 
                        checkpoint_data: Dict[str, Any]) -> None:
        """Save workflow checkpoint"""
        # TODO: Implement with journal persistence
        pass


# Export main classes
__all__ = [
    'ComponentType',
    'ExecutionContext', 
    'Component',
    'FunctionComponent',
    'ObjectComponent', 
    'FlowComponent',
    'StateManager',
    'FlowExecutor'
]