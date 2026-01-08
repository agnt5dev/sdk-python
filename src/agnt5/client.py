"""AGNT5 Client SDK for invoking components."""

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, Optional
from urllib.parse import urljoin

import httpx

if TYPE_CHECKING:
    from .client import RunError

# Environment variable for API key
AGNT5_API_KEY_ENV = "AGNT5_API_KEY"


@dataclass
class ReceivedEvent:
    """Event received from SSE stream.

    This is a simple container for events received via Server-Sent Events
    from the platform. It provides access to the event type and data payload
    without requiring the full typed event hierarchy used internally by the SDK.
    """

    event_type: str
    """Event type string (e.g., 'agent.started', 'lm.content_block.delta')."""

    data: Dict[str, Any]
    """Event data payload as a dictionary."""

    content_index: int = 0
    """Content block index for streaming events."""

    sequence: int = 0
    """Sequence number for ordering events."""


def _parse_error_response(error_data: Dict[str, Any], run_id: Optional[str] = None) -> "RunError":
    """Parse error response from platform and create RunError with structured fields.

    Args:
        error_data: Error response dict from platform (contains error, error_code, metadata, etc.)
        run_id: Optional run ID if not in error_data

    Returns:
        RunError with structured fields populated from the response
    """
    # Import here to avoid circular import
    from .client import RunError

    # Extract error message - handle both old format (string) and new format (dict)
    error = error_data.get("error", "Unknown error")
    if isinstance(error, dict):
        # New format: {"code": "...", "message": "..."}
        message = error.get("message", "Unknown error")
        error_code = error.get("code") or error_data.get("error_code")
    else:
        # Old format: error is a string
        message = error
        error_code = error_data.get("error_code")

    run_id = error_data.get("runId") or error_data.get("run_id") or run_id

    # Extract retry metadata if present
    metadata = error_data.get("metadata")
    attempts = None
    max_attempts = None

    if metadata:
        if isinstance(metadata, dict):
            attempts = metadata.get("attempts")
            max_attempts = metadata.get("max_attempts")
        elif isinstance(metadata, str):
            # Parse JSON string metadata
            try:
                parsed = json.loads(metadata)
                attempts = parsed.get("attempts")
                max_attempts = parsed.get("max_attempts")
                metadata = parsed
            except (json.JSONDecodeError, TypeError):
                metadata = {"raw": metadata}

    return RunError(
        message,
        run_id=run_id,
        error_code=error_code,
        attempts=attempts,
        max_attempts=max_attempts,
        metadata=metadata if isinstance(metadata, dict) else None,
    )


def _parse_sse_to_event(event_type_str: str, data: Dict[str, Any]) -> ReceivedEvent:
    """Convert SSE event type and data to ReceivedEvent.

    Args:
        event_type_str: The event type string from SSE (e.g., "agent.started")
        data: The parsed JSON data from the SSE data field

    Returns:
        ReceivedEvent with event_type string and data payload
    """
    return ReceivedEvent(
        event_type=event_type_str,
        data=data,
        content_index=data.get("index", 0),
        sequence=data.get("sequence", 0),
    )


class Client:
    """Client for invoking AGNT5 components.

    This client provides a simple interface for calling functions, workflows,
    and other components deployed on AGNT5.

    Example:
        ```python
        from agnt5 import Client

        # Local development (no auth needed)
        client = Client("http://localhost:34181")
        result = client.run("greet", {"name": "Alice"})
        print(result)  # {"message": "Hello, Alice!"}

        # Production with API key
        client = Client(
            gateway_url="https://api.agnt5.com",
            api_key="agnt5_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )

        # Or use AGNT5_API_KEY environment variable
        # export AGNT5_API_KEY=agnt5_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        client = Client(gateway_url="https://api.agnt5.com")
        ```
    """

    def __init__(
        self,
        gateway_url: str = "https://gw.agnt5.com",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """Initialize the AGNT5 client.

        Args:
            gateway_url: Base URL of the AGNT5 gateway (default: https://gw.agnt5.com)
            timeout: Request timeout in seconds (default: 30.0)
            api_key: Service key for authentication. If not provided, falls back to
                     AGNT5_API_KEY environment variable. Keys start with "agnt5_sk_".
        """
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout

        # Use provided api_key or fallback to environment variable
        self.api_key = api_key or os.environ.get(AGNT5_API_KEY_ENV)

        # Validate if the key starts with "agnt5_sk_"
        if self.api_key and not self.api_key.startswith("agnt5_sk_"):
            raise ValueError("Invalid API key format. Keys must start with 'agnt5_sk_'")

        self._client = httpx.Client(timeout=timeout)

    def _build_headers(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build request headers with authentication and optional session/user context.

        Args:
            session_id: Session identifier for multi-turn conversations
            user_id: User identifier for user-scoped memory

        Returns:
            Dictionary of HTTP headers
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        if session_id:
            headers["X-Session-ID"] = session_id
        if user_id:
            headers["X-User-ID"] = user_id

        return headers

    def run(
        self,
        component: str,
        input_data: Optional[Dict[str, Any]] = None,
        component_type: str = "function",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a component synchronously and wait for the result.

        This is a blocking call that waits for the component to complete execution.

        Args:
            component: Name of the component to execute
            input_data: Input data for the component (will be sent as JSON body)
            component_type: Type of component - "function", "workflow", "agent", "tool" (default: "function")
            session_id: Session identifier for multi-turn conversations (optional)
            user_id: User identifier for user-scoped memory (optional)
            timeout: Request timeout in seconds (optional, defaults to client timeout)
            headers: Additional HTTP headers to include in the request (optional, e.g., {"Idempotency-Key": "key"})

        Returns:
            Dictionary containing the component's output

        Raises:
            RunError: If the component execution fails
            httpx.HTTPError: If the HTTP request fails

        Example:
            ```python
            # Simple function call (default)
            result = client.run("greet", {"name": "Alice"})

            # Workflow execution (explicit)
            result = client.run("order_fulfillment", {"order_id": "123"}, component_type="workflow")

            # Multi-turn conversation with session
            result = client.run("chat", {"message": "Hello"}, session_id="session-123")

            # User-scoped memory
            result = client.run("assistant", {"message": "Help me"}, user_id="user-456")

            # No input data
            result = client.run("get_status")
            ```
        """
        if input_data is None:
            input_data = {}

        # Build URL with component type (plural form)
        url = urljoin(self.gateway_url + "/", f"v1/{component_type}s/{component}/run")

        # Build headers and merge with custom headers
        request_headers = self._build_headers(session_id=session_id, user_id=user_id)
        if headers:
            request_headers.update(headers)

        # Make request with auth and session headers
        response = self._client.post(
            url,
            json=input_data,
            headers=request_headers,
            timeout=timeout,
        )

        # Handle errors
        if response.status_code == 404:
            try:
                error_data = response.json()
                raise RunError(
                    error_data.get("error", "Component not found"),
                    run_id=error_data.get("runId"),
                )
            except ValueError:
                # JSON parsing failed
                raise RunError(f"Component '{component}' not found")

        if response.status_code == 503:
            error_data = response.json()
            raise RunError(
                f"Service unavailable: {error_data.get('error', 'Unknown error')}",
                run_id=error_data.get("runId"),
            )

        if response.status_code == 504:
            error_data = response.json()
            raise RunError(
                "Execution timeout",
                run_id=error_data.get("runId"),
            )

        # Handle 500 errors with our RunResponse format
        if response.status_code == 500:
            try:
                error_data = response.json()
                raise _parse_error_response(error_data)
            except ValueError:
                # JSON parsing failed, fall through to raise_for_status
                response.raise_for_status()
        else:
            # For other error codes, use standard HTTP error handling
            response.raise_for_status()

        # Parse response
        data = response.json()

        # Check execution status
        if data.get("status") == "failed":
            raise _parse_error_response(data)

        # Return output - extract from nested event structure
        # New format: data["result"]["output"]["output_data"]
        result = data.get("result", {})
        if result and isinstance(result, dict):
            output = result.get("output", {})
            if isinstance(output, dict) and "output_data" in output:
                return output["output_data"]
            return output
        # Fallback to old format
        return data.get("output", {})

    def submit(
        self,
        component: str,
        input_data: Optional[Dict[str, Any]] = None,
        component_type: str = "function",
    ) -> str:
        """Submit a component for async execution and return immediately.

        This is a non-blocking call that returns a run ID immediately.
        Use get_status() to check progress and get_result() to retrieve the output.

        Args:
            component: Name of the component to execute
            input_data: Input data for the component (will be sent as JSON body)
            component_type: Type of component - "function", "workflow", "agent", "tool" (default: "function")

        Returns:
            String containing the run ID

        Raises:
            httpx.HTTPError: If the HTTP request fails

        Example:
            ```python
            # Submit async function (default)
            run_id = client.submit("process_video", {"url": "https://..."})
            print(f"Submitted: {run_id}")

            # Submit workflow
            run_id = client.submit("order_fulfillment", {"order_id": "123"}, component_type="workflow")

            # Check status later
            status = client.get_status(run_id)
            if status["status"] == "completed":
                result = client.get_result(run_id)
            ```
        """
        if input_data is None:
            input_data = {}

        # Build URL with component type (plural form)
        url = urljoin(self.gateway_url + "/", f"v1/{component_type}s/{component}/submit")

        # Make request with auth headers
        response = self._client.post(
            url,
            json=input_data,
            headers=self._build_headers(),
        )

        # Handle errors
        response.raise_for_status()

        # Parse response and extract run ID
        # Submit endpoint uses snake_case "run_id" (not camelCase "runId")
        data = response.json()
        return data.get("run_id", "")

    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Get the current status of a run.

        Args:
            run_id: The run ID returned from submit()

        Returns:
            Dictionary containing status information:
            {
                "runId": "...",
                "status": "pending|running|completed|failed|cancelled",
                "submittedAt": 1234567890,
                "startedAt": 1234567891,  // optional
                "completedAt": 1234567892 // optional
            }

        Raises:
            httpx.HTTPError: If the HTTP request fails

        Example:
            ```python
            status = client.get_status(run_id)
            print(f"Status: {status['status']}")
            ```
        """
        url = urljoin(self.gateway_url + "/", f"v1/status/{run_id}")

        response = self._client.get(url, headers=self._build_headers())
        response.raise_for_status()

        return response.json()

    def get_result(self, run_id: str) -> Dict[str, Any]:
        """Get the result of a completed run.

        This will raise an error if the run is not yet complete.

        Args:
            run_id: The run ID returned from submit()

        Returns:
            Dictionary containing the component's output

        Raises:
            RunError: If the run failed or is not yet complete
            httpx.HTTPError: If the HTTP request fails

        Example:
            ```python
            try:
                result = client.get_result(run_id)
                print(result)
            except RunError as e:
                if "not complete" in str(e):
                    print("Run is still in progress")
                else:
                    print(f"Run failed: {e}")
            ```
        """
        url = urljoin(self.gateway_url + "/", f"v1/result/{run_id}")

        response = self._client.get(url, headers=self._build_headers())

        # Handle 404 - run not complete or not found
        if response.status_code == 404:
            error_data = response.json()
            error_msg = error_data.get("error", "Run not found or not complete")
            current_status = error_data.get("status", "unknown")
            raise RunError(f"{error_msg} (status: {current_status})", run_id=run_id)

        # Handle other errors
        response.raise_for_status()

        # Parse response
        data = response.json()

        # Check if run failed
        if data.get("status") == "failed":
            raise _parse_error_response(data, run_id=run_id)

        # Return output - extract from nested event structure
        # New format: data["result"]["output"]["output_data"]
        result = data.get("result", {})
        if result and isinstance(result, dict):
            output = result.get("output", {})
            if isinstance(output, dict) and "output_data" in output:
                return output["output_data"]
            return output
        # Fallback to old format
        return data.get("output", {})

    def wait_for_result(
        self,
        run_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> Dict[str, Any]:
        """Wait for a run to complete and return the result.

        This polls the status endpoint until the run completes or times out.

        Args:
            run_id: The run ID returned from submit()
            timeout: Maximum time to wait in seconds (default: 300)
            poll_interval: How often to check status in seconds (default: 1.0)

        Returns:
            Dictionary containing the component's output

        Raises:
            RunError: If the run fails or times out
            httpx.HTTPError: If the HTTP request fails

        Example:
            ```python
            # Submit and wait for result
            run_id = client.submit("long_task", {"data": "..."})
            try:
                result = client.wait_for_result(run_id, timeout=600)
                print(result)
            except RunError as e:
                print(f"Failed: {e}")
            ```
        """
        import time

        start_time = time.time()

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise RunError(
                    f"Timeout waiting for run to complete after {timeout}s",
                    run_id=run_id,
                )

            # Get current status
            status = self.get_status(run_id)
            current_status = status.get("status", "")

            # Check if complete
            if current_status in ("completed", "failed", "cancelled"):
                # Get result (will raise if failed)
                return self.get_result(run_id)

            # Wait before next poll
            time.sleep(poll_interval)

    def stream(
        self,
        component: str,
        input_data: Optional[Dict[str, Any]] = None,
        component_type: str = "function",
    ):
        """Stream responses from a component using Server-Sent Events (SSE).

        This method yields chunks as they arrive from the component.
        Perfect for LLM token streaming and incremental responses.

        Args:
            component: Name of the component to execute
            input_data: Input data for the component (will be sent as JSON body)
            component_type: Type of component - "function", "workflow", "agent", "tool" (default: "function")

        Yields:
            String chunks as they arrive from the component

        Raises:
            RunError: If the component execution fails
            httpx.HTTPError: If the HTTP request fails

        Example:
            ```python
            # Stream LLM tokens
            for chunk in client.stream("generate_text", {"prompt": "Write a story"}):
                print(chunk, end="", flush=True)
            ```
        """
        if input_data is None:
            input_data = {}

        # Build URL with component type (plural form)
        url = urljoin(self.gateway_url + "/", f"v1/{component_type}s/{component}/stream")

        # Use streaming request with auth headers
        with self._client.stream(
            "POST",
            url,
            json=input_data,
            headers=self._build_headers(),
            timeout=300.0,  # 5 minute timeout for streaming
        ) as response:
            # Check for errors
            if response.status_code != 200:
                # For streaming responses, we can't read the full text
                # Just raise an HTTP error
                raise RunError(
                    f"HTTP {response.status_code}: Streaming request failed",
                    run_id=None,
                )

            # Parse SSE stream
            current_event = None
            for line in response.iter_lines():
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith(":"):
                    continue

                # Parse event type: "event: output.delta"
                if line.startswith("event: "):
                    current_event = line[7:]  # Remove "event: " prefix
                    continue

                # Parse SSE format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    try:
                        data = json.loads(data_str)

                        # Check for completion
                        if data.get("done") or current_event == "done":
                            return

                        # Check for error
                        if "error" in data:
                            # Use helper to properly parse error structure
                            raise _parse_error_response(data, run_id=data.get("runId") or data.get("run_id"))

                        # Yield chunk from output.delta events
                        if current_event == "output.delta":
                            # Try different content field formats
                            if "content" in data:
                                yield data["content"]
                            elif "output_data" in data:
                                # output_data is proper JSON (string, number, object, etc.)
                                output = data["output_data"]
                                if isinstance(output, str):
                                    yield output
                                elif output is not None:
                                    # For non-string types, yield JSON string representation
                                    yield json.dumps(output)
                        # Also support legacy "chunk" format
                        elif "chunk" in data:
                            yield data["chunk"]

                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue

    def stream_events(
        self,
        component: str,
        input_data: Optional[Dict[str, Any]] = None,
        component_type: str = "function",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> Iterator[ReceivedEvent]:
        """Stream events from a component execution.

        This method yields ReceivedEvent objects as they arrive from the component,
        providing access to the event taxonomy including agent lifecycle,
        LM streaming, tool calls, and workflow events.

        Args:
            component: Name of the component to execute
            input_data: Input data for the component (will be sent as JSON body)
            component_type: Type of component - "function", "workflow", "agent", "tool"
            session_id: Session identifier for multi-turn conversations (optional)
            user_id: User identifier for user-scoped memory (optional)
            timeout: Stream timeout in seconds (default: 300.0 / 5 minutes)

        Yields:
            ReceivedEvent objects as they arrive from the stream

        Raises:
            RunError: If the component execution fails
            httpx.HTTPError: If the HTTP request fails

        Example:
            ```python
            from agnt5 import Client

            client = Client()

            # Stream agent events
            for event in client.stream_events("my_agent", {"message": "Hi"}, "agent"):
                if event.event_type == "agent.started":
                    print(f"Agent started: {event.data['agent_name']}")
                elif event.event_type == "lm.content_block.delta":
                    print(event.data['content'], end='', flush=True)
                elif event.event_type == "agent.completed":
                    print(f"\\nDone: {event.data['output']}")
            ```
        """
        if timeout <= 0:
            raise ValueError("timeout must be a positive number")

        if input_data is None:
            input_data = {}

        # Build URL with component type (using streaming endpoint, plural form)
        url = urljoin(self.gateway_url + "/", f"v1/{component_type}s/{component}/stream")

        # Use streaming request with auth and session headers
        with self._client.stream(
            "POST",
            url,
            json=input_data,
            headers=self._build_headers(session_id=session_id, user_id=user_id),
            timeout=timeout,
        ) as response:
            # Check for errors
            if response.status_code != 200:
                # Try to get error details from response body
                try:
                    error_body = response.read().decode("utf-8")
                    error_data = json.loads(error_body)
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                    run_id = error_data.get("runId")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    error_msg = f"HTTP {response.status_code}: Streaming request failed"
                    run_id = None
                raise RunError(error_msg, run_id=run_id)

            # Parse SSE stream
            current_event_type: Optional[str] = None
            for line in response.iter_lines():
                line = line.strip()

                # Skip empty lines and comments (keep-alive)
                if not line or line.startswith(":"):
                    continue

                # Parse event type: "event: agent.started"
                if line.startswith("event: "):
                    current_event_type = line[7:]  # Remove "event: " prefix
                    continue

                # Parse SSE data: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    try:
                        data = json.loads(data_str)

                        # Check for completion signal
                        if data.get("done") or current_event_type == "done":
                            return

                        # Check for error event
                        if current_event_type == "error" or "error" in data:
                            # Use helper to properly parse error structure
                            raise _parse_error_response(data, run_id=data.get("runId") or data.get("run_id"))

                        # Yield typed Event object
                        if current_event_type:
                            yield _parse_sse_to_event(current_event_type, data)

                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue

    def entity(self, entity_type: str, key: str) -> "EntityProxy":
        """Get a proxy for calling methods on a durable entity.

        This provides a fluent API for entity method invocations with key-based routing.

        Args:
            entity_type: The entity class name (e.g., "Counter", "ShoppingCart")
            key: The entity instance key (e.g., "user-123", "cart-alice")

        Returns:
            EntityProxy that allows method calls on the entity

        Example:
            ```python
            # Call entity method
            result = client.entity("Counter", "user-123").increment(amount=5)
            print(result)  # 5

            # Shopping cart
            result = client.entity("ShoppingCart", "user-alice").add_item(
                item_id="item-123",
                quantity=2,
                price=29.99
            )
            ```
        """
        return EntityProxy(self, entity_type, key)

    def workflow(self, workflow_name: str) -> "WorkflowProxy":
        """Get a proxy for invoking a workflow with fluent API.

        This provides a convenient API for workflow invocations, including
        a chat() method for multi-turn conversation workflows.

        Args:
            workflow_name: Name of the workflow to invoke

        Returns:
            WorkflowProxy that provides workflow-specific methods

        Example:
            ```python
            # Standard workflow execution
            result = client.workflow("order_process").run(order_id="123")

            # Chat workflow with session
            response = client.workflow("support_bot").chat(
                message="My order hasn't arrived",
                session_id="user-123",
            )

            # Continue conversation
            response = client.workflow("support_bot").chat(
                message="Can you track it?",
                session_id="user-123",
            )
            ```
        """
        return WorkflowProxy(self, workflow_name)

    def session(self, session_type: str, key: str) -> "SessionProxy":
        """Get a proxy for a session entity (OpenAI/ADK-style API).

        This is a convenience wrapper around entity() specifically for SessionEntity subclasses,
        providing a familiar API for developers coming from OpenAI Agents SDK or Google ADK.

        Args:
            session_type: The session entity class name (e.g., "Conversation", "ChatSession")
            key: The session instance key (typically user ID or session ID)

        Returns:
            SessionProxy that provides session-specific methods

        Example:
            ```python
            # Create a conversation session
            session = client.session("Conversation", "user-alice")

            # Chat with the session
            response = session.chat("Hello! How are you?")
            print(response)

            # Get conversation history
            history = session.get_history()
            for msg in history:
                print(f"{msg['role']}: {msg['content']}")
            ```
        """
        return SessionProxy(self, session_type, key)

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class EntityProxy:
    """Proxy for calling methods on a durable entity instance.

    This class enables fluent method calls on entities using Python's
    attribute access. Any method call is translated to an HTTP request
    to /entity/:type/:key/:method.

    Example:
        ```python
        counter = client.entity("Counter", "user-123")
        result = counter.increment(amount=5)  # Calls /entity/Counter/user-123/increment
        ```
    """

    def __init__(self, client: "Client", entity_type: str, key: str):
        """Initialize entity proxy.

        Args:
            client: The AGNT5 client instance
            entity_type: The entity class name
            key: The entity instance key
        """
        self._client = client
        self._entity_type = entity_type
        self._key = key

    def __getattr__(self, method_name: str):
        """Dynamic method lookup that creates entity method callers.

        Args:
            method_name: The entity method to call

        Returns:
            Callable that executes the entity method
        """

        def method_caller(*args, **kwargs) -> Any:
            """Call an entity method with the given parameters.

            Args:
                *args: Positional arguments (not recommended, use kwargs)
                **kwargs: Method parameters as keyword arguments

            Returns:
                The method's return value

            Raises:
                RunError: If the method execution fails
                ValueError: If both positional and keyword arguments are provided
            """
            # Convert positional args to kwargs if provided
            if args and kwargs:
                raise ValueError(
                    f"Cannot mix positional and keyword arguments when calling entity method '{method_name}'. "
                    "Please use keyword arguments only."
                )

            # If positional args provided, we can't convert them without knowing parameter names
            # Raise helpful error
            if args:
                raise ValueError(
                    f"Entity method '{method_name}' requires keyword arguments, but got {len(args)} positional arguments. "
                    f"Example: .{method_name}(param1=value1, param2=value2)"
                )

            # Build URL: /v1/entity/:entityType/:key/:method
            url = urljoin(
                self._client.gateway_url + "/",
                f"v1/entity/{self._entity_type}/{self._key}/{method_name}",
            )

            # Make request with method parameters as JSON body and auth headers
            response = self._client._client.post(
                url,
                json=kwargs,
                headers=self._client._build_headers(),
            )

            # Handle errors
            if response.status_code == 504:
                error_data = response.json()
                raise RunError(
                    "Execution timeout",
                    run_id=error_data.get("run_id"),
                )

            if response.status_code == 500:
                try:
                    error_data = response.json()
                    raise _parse_error_response(error_data)
                except ValueError:
                    response.raise_for_status()
            else:
                response.raise_for_status()

            # Parse response
            data = response.json()

            # Check execution status
            if data.get("status") == "failed":
                raise _parse_error_response(data)

            # Return output - extract from nested event structure
            # New format: data["result"]["output"]["output_data"]
            result = data.get("result", {})
            if result and isinstance(result, dict):
                output = result.get("output", {})
                if isinstance(output, dict) and "output_data" in output:
                    return output["output_data"]
                return output
            # Fallback to old format
            return data.get("output")

        return method_caller


class SessionProxy(EntityProxy):
    """Proxy for session entities with conversation-specific helper methods.

    This extends EntityProxy to provide familiar APIs for session-based
    conversations, similar to OpenAI Agents SDK and Google ADK.

    Example:
        ```python
        # Create a session
        session = client.session("Conversation", "user-alice")

        # Chat
        response = session.chat("Tell me about AI")

        # Get history
        history = session.get_history()
        ```
    """

    def chat(self, message: str, **kwargs) -> str:
        """Send a message to the conversation session.

        This is a convenience method that calls the `chat` method on the
        underlying SessionEntity and returns just the response text.

        Args:
            message: The user's message
            **kwargs: Additional parameters to pass to the chat method

        Returns:
            The assistant's response as a string

        Example:
            ```python
            response = session.chat("What is the weather today?")
            print(response)
            ```
        """
        # Call the chat method via the entity proxy
        result = self.__getattr__("chat")(message=message, **kwargs)

        # SessionEntity.chat() returns a dict with 'response' key
        if isinstance(result, dict) and "response" in result:
            return result["response"]

        # If it's already a string, return as-is
        return str(result)

    def get_history(self) -> list:
        """Get the conversation history for this session.

        Returns:
            List of message dictionaries with 'role' and 'content' keys

        Example:
            ```python
            history = session.get_history()
            for msg in history:
                print(f"{msg['role']}: {msg['content']}")
            ```
        """
        return self.__getattr__("get_history")()

    def add_message(self, role: str, content: str) -> dict:
        """Add a message to the conversation history.

        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content

        Returns:
            Dictionary confirming the message was added

        Example:
            ```python
            session.add_message("system", "You are a helpful assistant")
            session.add_message("user", "Hello!")
            ```
        """
        return self.__getattr__("add_message")(role=role, content=content)

    def clear_history(self) -> dict:
        """Clear the conversation history for this session.

        Returns:
            Dictionary confirming the history was cleared

        Example:
            ```python
            session.clear_history()
            ```
        """
        return self.__getattr__("clear_history")()


class WorkflowProxy:
    """Proxy for invoking workflows with a fluent API.

    Provides convenient methods for workflow execution, including
    a chat() method for multi-turn conversation workflows.

    Example:
        ```python
        # Standard workflow
        result = client.workflow("order_process").run(order_id="123")

        # Chat workflow
        response = client.workflow("support_bot").chat(
            message="Help me",
            session_id="user-123",
        )
        ```
    """

    def __init__(self, client: "Client", workflow_name: str):
        """Initialize workflow proxy.

        Args:
            client: The AGNT5 client instance
            workflow_name: Name of the workflow
        """
        self._client = client
        self._workflow_name = workflow_name

    def run(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the workflow synchronously.

        Args:
            session_id: Session identifier for multi-turn workflows (optional)
            user_id: User identifier for user-scoped memory (optional)
            **kwargs: Input parameters for the workflow

        Returns:
            Dictionary containing the workflow's output

        Example:
            ```python
            result = client.workflow("order_process").run(
                order_id="123",
                customer_id="cust-456",
            )
            ```
        """
        return self._client.run(
            component=self._workflow_name,
            input_data=kwargs,
            component_type="workflow",
            session_id=session_id,
            user_id=user_id,
        )

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send a message to a chat-enabled workflow.

        This is a convenience method for multi-turn conversation workflows.
        The message is passed as the 'message' input parameter.

        Args:
            message: The user's message
            session_id: Session identifier for conversation continuity (recommended)
            user_id: User identifier for user-scoped memory (optional)
            **kwargs: Additional input parameters for the workflow

        Returns:
            Dictionary containing the workflow's response (typically has 'response' key)

        Example:
            ```python
            # First message
            result = client.workflow("support_bot").chat(
                message="My order hasn't arrived",
                session_id="session-123",
            )
            print(result.get("response"))

            # Continue conversation
            result = client.workflow("support_bot").chat(
                message="Can you track it?",
                session_id="session-123",
            )
            ```
        """
        # Merge message into kwargs
        input_data = {"message": message, **kwargs}

        return self._client.run(
            component=self._workflow_name,
            input_data=input_data,
            component_type="workflow",
            session_id=session_id,
            user_id=user_id,
        )

    def stream_events(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: float = 300.0,
        **kwargs,
    ) -> Iterator[ReceivedEvent]:
        """Stream events from workflow execution.

        This method yields ReceivedEvent objects as they arrive from the workflow,
        including nested events from agents and functions called within the workflow.

        Args:
            session_id: Session identifier for multi-turn workflows (optional)
            user_id: User identifier for user-scoped memory (optional)
            timeout: Stream timeout in seconds (default: 300.0 / 5 minutes)
            **kwargs: Input parameters for the workflow

        Yields:
            ReceivedEvent objects as they arrive from the stream

        Example:
            ```python
            from agnt5 import Client

            # Stream workflow events
            for event in client.workflow("research_workflow").stream_events(query="AI"):
                if event.event_type == "workflow.step.started":
                    print(f"Step started: {event.data.get('step_name')}")
                elif event.event_type == "lm.content_block.delta":
                    print(event.data['content'], end='', flush=True)
                elif event.event_type == "workflow.step.completed":
                    print(f"\\nStep done: {event.data.get('step_name')}")
            ```
        """
        return self._client.stream_events(
            component=self._workflow_name,
            input_data=kwargs,
            component_type="workflow",
            session_id=session_id,
            user_id=user_id,
            timeout=timeout,
        )

    def submit(self, **kwargs) -> str:
        """Submit the workflow for async execution.

        Args:
            **kwargs: Input parameters for the workflow

        Returns:
            Run ID for tracking the execution

        Example:
            ```python
            run_id = client.workflow("long_process").submit(data="...")
            # Check status later
            status = client.get_status(run_id)
            ```
        """
        return self._client.submit(
            component=self._workflow_name,
            input_data=kwargs,
            component_type="workflow",
        )


class AsyncClient:
    """Async client for invoking AGNT5 components.

    This client provides an async interface for calling functions, workflows,
    and other components deployed on AGNT5. Use this when you need to stream
    events in an async context or integrate with async frameworks.

    Example:
        ```python
        import asyncio
        from agnt5 import AsyncClient

        async def main():
            async with AsyncClient() as client:
                # Stream agent events asynchronously
                async for event in client.stream_events("my_agent", {"msg": "Hi"}, "agent"):
                    if event.event_type == "lm.content_block.delta":
                        print(event.data['content'], end='', flush=True)

        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        gateway_url: str = "http://localhost:34181",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """Initialize the async AGNT5 client.

        Args:
            gateway_url: Base URL of the AGNT5 gateway (default: http://localhost:34181)
            timeout: Request timeout in seconds (default: 30.0)
            api_key: Service key for authentication. If not provided, falls back to
                     AGNT5_API_KEY environment variable. Keys start with "agnt5_sk_".
        """
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
        # Use provided api_key or fallback to environment variable
        self.api_key = api_key or os.environ.get(AGNT5_API_KEY_ENV)
        self._client: Optional[httpx.AsyncClient] = None

    def _build_headers(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build request headers with authentication and optional session/user context.

        Args:
            session_id: Session identifier for multi-turn conversations
            user_id: User identifier for user-scoped memory

        Returns:
            Dictionary of HTTP headers
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        if session_id:
            headers["X-Session-ID"] = session_id
        if user_id:
            headers["X-User-ID"] = user_id
        return headers

    async def __aenter__(self) -> "AsyncClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure async client is available."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def run(
        self,
        component: str,
        input_data: Optional[Dict[str, Any]] = None,
        component_type: str = "function",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a component asynchronously and wait for the result.

        Args:
            component: Name of the component to execute
            input_data: Input data for the component
            component_type: Type of component - "function", "workflow", "agent", "tool"
            session_id: Session identifier for multi-turn conversations
            user_id: User identifier for user-scoped memory

        Returns:
            Dictionary containing the component's output

        Raises:
            RunError: If the component execution fails
            httpx.HTTPError: If the HTTP request fails
        """
        if input_data is None:
            input_data = {}

        client = await self._ensure_client()
        url = urljoin(self.gateway_url + "/", f"v1/{component_type}s/{component}/run")

        response = await client.post(
            url,
            json=input_data,
            headers=self._build_headers(session_id=session_id, user_id=user_id),
        )

        if response.status_code == 404:
            try:
                error_data = response.json()
                raise RunError(
                    error_data.get("error", "Component not found"),
                    run_id=error_data.get("runId"),
                )
            except ValueError:
                raise RunError(f"Component '{component}' not found")

        if response.status_code in (500, 503, 504):
            try:
                error_data = response.json()
                raise _parse_error_response(error_data)
            except ValueError:
                response.raise_for_status()
        else:
            response.raise_for_status()

        data = response.json()
        if data.get("status") == "failed":
            raise _parse_error_response(data)

        # Return output - extract from nested event structure
        # New format: data["result"]["output"]["output_data"]
        result = data.get("result", {})
        if result and isinstance(result, dict):
            output = result.get("output", {})
            if isinstance(output, dict) and "output_data" in output:
                return output["output_data"]
            return output
        # Fallback to old format
        return data.get("output", {})

    async def stream_events(
        self,
        component: str,
        input_data: Optional[Dict[str, Any]] = None,
        component_type: str = "function",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> AsyncIterator[ReceivedEvent]:
        """Async stream events from a component execution.

        This method yields ReceivedEvent objects as they arrive from the component,
        providing access to the event taxonomy including agent lifecycle,
        LM streaming, tool calls, and workflow events.

        Args:
            component: Name of the component to execute
            input_data: Input data for the component
            component_type: Type of component - "function", "workflow", "agent", "tool"
            session_id: Session identifier for multi-turn conversations
            user_id: User identifier for user-scoped memory
            timeout: Stream timeout in seconds (default: 300.0 / 5 minutes)

        Yields:
            ReceivedEvent objects as they arrive from the stream

        Raises:
            RunError: If the component execution fails
            httpx.HTTPError: If the HTTP request fails

        Example:
            ```python
            async with AsyncClient() as client:
                async for event in client.stream_events("my_agent", {"msg": "Hi"}, "agent"):
                    if event.event_type == "agent.started":
                        print(f"Agent started: {event.data['agent_name']}")
                    elif event.event_type == "lm.content_block.delta":
                        print(event.data['content'], end='', flush=True)
            ```
        """
        if timeout <= 0:
            raise ValueError("timeout must be a positive number")

        if input_data is None:
            input_data = {}

        client = await self._ensure_client()
        url = urljoin(self.gateway_url + "/", f"v1/{component_type}s/{component}/stream")

        async with client.stream(
            "POST",
            url,
            json=input_data,
            headers=self._build_headers(session_id=session_id, user_id=user_id),
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                # Try to get error details from response body
                try:
                    error_body = (await response.aread()).decode("utf-8")
                    error_data = json.loads(error_body)
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                    run_id = error_data.get("runId")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    error_msg = f"HTTP {response.status_code}: Streaming request failed"
                    run_id = None
                raise RunError(error_msg, run_id=run_id)

            current_event_type: Optional[str] = None
            async for line in response.aiter_lines():
                line = line.strip()

                # Skip empty lines and comments (keep-alive)
                if not line or line.startswith(":"):
                    continue

                # Parse event type: "event: agent.started"
                if line.startswith("event: "):
                    current_event_type = line[7:]
                    continue

                # Parse SSE data: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]

                    try:
                        data = json.loads(data_str)

                        # Check for completion signal
                        if data.get("done") or current_event_type == "done":
                            return

                        # Check for error event
                        if current_event_type == "error" or "error" in data:
                            # Use helper to properly parse error structure
                            raise _parse_error_response(data, run_id=data.get("runId") or data.get("run_id"))

                        # Yield typed Event object
                        if current_event_type:
                            yield _parse_sse_to_event(current_event_type, data)

                    except json.JSONDecodeError:
                        continue

    async def submit(
        self,
        component: str,
        input_data: Optional[Dict[str, Any]] = None,
        component_type: str = "function",
    ) -> str:
        """Submit a component for async execution and return immediately.

        Args:
            component: Name of the component to execute
            input_data: Input data for the component
            component_type: Type of component

        Returns:
            String containing the run ID
        """
        if input_data is None:
            input_data = {}

        client = await self._ensure_client()
        url = urljoin(self.gateway_url + "/", f"v1/{component_type}s/{component}/submit")

        response = await client.post(
            url,
            json=input_data,
            headers=self._build_headers(),
        )
        response.raise_for_status()

        # Submit endpoint uses snake_case "run_id" (not camelCase "runId")
        data = response.json()
        return data.get("run_id", "")

    async def get_status(self, run_id: str) -> Dict[str, Any]:
        """Get the current status of a run.

        Args:
            run_id: The run ID returned from submit()

        Returns:
            Dictionary containing status information
        """
        client = await self._ensure_client()
        url = urljoin(self.gateway_url + "/", f"v1/status/{run_id}")

        response = await client.get(url, headers=self._build_headers())
        response.raise_for_status()

        return response.json()

    async def get_result(self, run_id: str) -> Dict[str, Any]:
        """Get the result of a completed run.

        Args:
            run_id: The run ID returned from submit()

        Returns:
            Dictionary containing the component's output

        Raises:
            RunError: If the run failed or is not yet complete
        """
        client = await self._ensure_client()
        url = urljoin(self.gateway_url + "/", f"v1/result/{run_id}")

        response = await client.get(url, headers=self._build_headers())

        if response.status_code == 404:
            error_data = response.json()
            error_msg = error_data.get("error", "Run not found or not complete")
            current_status = error_data.get("status", "unknown")
            raise RunError(f"{error_msg} (status: {current_status})", run_id=run_id)

        response.raise_for_status()
        data = response.json()

        if data.get("status") == "failed":
            raise _parse_error_response(data, run_id=run_id)

        # Return output - extract from nested event structure
        # New format: data["result"]["output"]["output_data"]
        result = data.get("result", {})
        if result and isinstance(result, dict):
            output = result.get("output", {})
            if isinstance(output, dict) and "output_data" in output:
                return output["output_data"]
            return output
        # Fallback to old format
        return data.get("output", {})


class RunError(Exception):
    """Raised when a component run fails on AGNT5.

    Attributes:
        message: Error message describing what went wrong
        run_id: The unique run ID associated with this execution (if available)
        error_code: Structured error code (e.g., "EXECUTION_FAILED", "GRPC_ERROR")
        attempts: Number of execution attempts made (1-indexed)
        max_attempts: Maximum attempts configured for this component
        metadata: Full metadata dict from platform response
    """

    def __init__(
        self,
        message: str,
        run_id: Optional[str] = None,
        error_code: Optional[str] = None,
        attempts: Optional[int] = None,
        max_attempts: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.run_id = run_id
        self.error_code = error_code
        self.attempts = attempts
        self.max_attempts = max_attempts
        self.metadata = metadata or {}

    @property
    def was_retried(self) -> bool:
        """Returns True if execution was retried at least once."""
        return self.attempts is not None and self.attempts > 1

    @property
    def exhausted_retries(self) -> bool:
        """Returns True if all retry attempts were exhausted."""
        if self.attempts is None or self.max_attempts is None:
            return False
        return self.attempts >= self.max_attempts

    def __str__(self):
        parts = [self.message]
        if self.run_id:
            parts.append(f"run_id: {self.run_id}")
        if self.attempts is not None and self.max_attempts is not None:
            parts.append(f"attempts: {self.attempts}/{self.max_attempts}")
        if self.error_code:
            parts.append(f"error_code: {self.error_code}")
        if len(parts) > 1:
            return f"{parts[0]} ({', '.join(parts[1:])})"
        return self.message
