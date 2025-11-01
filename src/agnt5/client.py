"""AGNT5 Client SDK for invoking components."""

import json
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx


class Client:
    """Client for invoking AGNT5 components.

    This client provides a simple interface for calling functions, workflows,
    and other components deployed on AGNT5.

    Example:
        ```python
        from agnt5 import Client

        client = Client("http://localhost:34181")
        result = client.run("greet", {"name": "Alice"})
        print(result)  # {"message": "Hello, Alice!"}
        ```
    """

    def __init__(
        self,
        gateway_url: str = "http://localhost:34181",
        timeout: float = 30.0,
    ):
        """Initialize the AGNT5 client.

        Args:
            gateway_url: Base URL of the AGNT5 gateway (default: http://localhost:34181)
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def run(
        self,
        component: str,
        input_data: Optional[Dict[str, Any]] = None,
        component_type: str = "function",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a component synchronously and wait for the result.

        This is a blocking call that waits for the component to complete execution.

        Args:
            component: Name of the component to execute
            input_data: Input data for the component (will be sent as JSON body)
            component_type: Type of component - "function", "workflow", "agent", "tool" (default: "function")
            session_id: Session identifier for multi-turn conversations (optional)
            user_id: User identifier for user-scoped memory (optional)

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

        # Build URL with component type
        url = urljoin(self.gateway_url + "/", f"v1/run/{component_type}/{component}")

        # Build headers with memory scoping identifiers
        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-ID"] = session_id
        if user_id:
            headers["X-User-ID"] = user_id

        # Make request
        response = self._client.post(
            url,
            json=input_data,
            headers=headers,
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
                raise RunError(
                    error_data.get("error", "Unknown error"),
                    run_id=error_data.get("runId"),
                )
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
            raise RunError(
                data.get("error", "Unknown error"),
                run_id=data.get("runId"),
            )

        # Return output
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

        # Build URL with component type
        url = urljoin(self.gateway_url + "/", f"v1/submit/{component_type}/{component}")

        # Make request
        response = self._client.post(
            url,
            json=input_data,
            headers={"Content-Type": "application/json"},
        )

        # Handle errors
        response.raise_for_status()

        # Parse response and extract run ID
        data = response.json()
        return data.get("runId", "")

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

        response = self._client.get(url)
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

        response = self._client.get(url)

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
            raise RunError(
                data.get("error", "Unknown error"),
                run_id=run_id,
            )

        # Return output
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
    ):
        """Stream responses from a component using Server-Sent Events (SSE).

        This method yields chunks as they arrive from the component.
        Perfect for LLM token streaming and incremental responses.

        Args:
            component: Name of the component to execute
            input_data: Input data for the component (will be sent as JSON body)

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

        # Build URL
        url = urljoin(self.gateway_url + "/", f"v1/stream/{component}")

        # Use streaming request
        with self._client.stream(
            "POST",
            url,
            json=input_data,
            headers={"Content-Type": "application/json"},
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
            for line in response.iter_lines():
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith(":"):
                    continue

                # Parse SSE format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    try:
                        data = json.loads(data_str)

                        # Check for completion
                        if data.get("done"):
                            return

                        # Check for error
                        if "error" in data:
                            raise RunError(
                                data.get("error"),
                                run_id=data.get("runId"),
                            )

                        # Yield chunk
                        if "chunk" in data:
                            yield data["chunk"]

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

            # Make request with method parameters as JSON body
            response = self._client._client.post(
                url,
                json=kwargs,
                headers={"Content-Type": "application/json"},
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
                    raise RunError(
                        error_data.get("error", "Unknown error"),
                        run_id=error_data.get("run_id"),
                    )
                except ValueError:
                    response.raise_for_status()
            else:
                response.raise_for_status()

            # Parse response
            data = response.json()

            # Check execution status
            if data.get("status") == "failed":
                raise RunError(
                    data.get("error", "Unknown error"),
                    run_id=data.get("run_id"),
                )

            # Return output
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


class RunError(Exception):
    """Raised when a component run fails on AGNT5.

    Attributes:
        message: Error message describing what went wrong
        run_id: The unique run ID associated with this execution (if available)
    """

    def __init__(self, message: str, run_id: Optional[str] = None):
        super().__init__(message)
        self.run_id = run_id
        self.message = message

    def __str__(self):
        if self.run_id:
            return f"{self.message} (run_id: {self.run_id})"
        return self.message
