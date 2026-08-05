"""AGNT5 SDK exceptions and error types."""

from enum import Enum
from typing import Dict, List, Optional


class AGNT5Error(Exception):
    """Base exception for all AGNT5 SDK errors."""

    pass


class ConfigurationError(AGNT5Error):
    """Raised when SDK configuration is invalid."""

    pass


class ExecutionError(AGNT5Error):
    """Raised when function or workflow execution fails."""

    pass


class RetryError(ExecutionError):
    """Raised when a function exceeds maximum retry attempts.

    .. deprecated::
        RetryError is deprecated and will be removed in a future version.
        Retry logic is now handled by the platform (Execution Engine).
        Functions are executed once by the SDK; the platform orchestrates retries.
    """

    def __init__(self, message: str, attempts: int, last_error: Exception) -> None:
        import warnings

        warnings.warn(
            "RetryError is deprecated. Retry logic is now handled by the platform. "
            "Functions are executed once by the SDK; the platform orchestrates retries.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class StateError(AGNT5Error):
    """Raised when state operations fail."""

    pass


class CheckpointError(AGNT5Error):
    """Raised when checkpoint operations fail."""

    pass


class ActivationErrorCode(str, Enum):
    """Stable durable-activation error codes shared by every AGNT5 SDK."""

    DURABILITY_UNAVAILABLE = "DURABILITY_UNAVAILABLE"
    NON_DETERMINISTIC_REPLAY = "NON_DETERMINISTIC_REPLAY"
    STALE_AUTHORITY = "STALE_AUTHORITY"
    CANCELLED = "CANCELLED"
    CONTENDED = "CONTENDED"
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"
    PAYLOAD_CONFLICT = "PAYLOAD_CONFLICT"
    ILLEGAL_TRANSITION = "ILLEGAL_TRANSITION"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    REFERENCE_REQUIRED = "REFERENCE_REQUIRED"
    STATE_VERSION_CONFLICT = "STATE_VERSION_CONFLICT"


class ActivationError(AGNT5Error):
    """A correctness failure at the durable-activation boundary."""

    def __init__(
        self,
        code: ActivationErrorCode,
        message: str,
        *,
        activation_id: str = "",
        attempt: int = 0,
    ) -> None:
        self.code = code
        self.activation_id = activation_id
        self.attempt = attempt
        identity = ""
        if activation_id:
            identity = f" activation={activation_id} attempt={attempt}"
        super().__init__(f"durable activation {code.value}{identity}: {message}")


class NotImplementedError(AGNT5Error):
    """Raised when a feature is not yet implemented."""

    pass


class WaitingForUserInputException(BaseException):
    """Raised when workflow needs to pause for user input.

    This exception is used internally by ctx.wait_for_user() to signal
    that a workflow execution should pause and wait for user input.

    IMPORTANT: This inherits from BaseException (not Exception) to prevent
    accidental catching by broad `except Exception:` blocks. This is the same
    pattern Python uses for KeyboardInterrupt and SystemExit.

    The platform catches this exception and:
    1. Saves the workflow checkpoint state
    2. Returns awaiting_user_input status to the client
    3. Presents the question and options to the user
    4. Resumes execution when user responds

    Attributes:
        question: The question to ask the user
        input_type: Type of input ("text", "approval", "select", or "multiselect")
        options: List of options for approval/select/multiselect inputs
        checkpoint_state: Current workflow state for resume
        pause_index: The index of this pause point (for multi-step HITL)
        step_name: The name of the wait_for_user step (e.g., "wait_for_user_0")
        step_correlation_id: Correlation ID for the step events
        agent_context: Optional agent execution state for agent-level HITL
            Contains: agent_name, iteration, messages, tool_results, pending_tool_call, etc.
        allow_custom: Whether to allow a free-text "Something else" option
        skippable: Whether the user can skip this input
    """

    def __init__(
        self,
        question: str,
        input_type: str,
        options: Optional[List[Dict]],
        checkpoint_state: Dict,
        pause_index: int = 0,
        agent_context: Optional[Dict] = None,
        step_name: Optional[str] = None,
        step_correlation_id: Optional[str] = None,
        allow_custom: bool = False,
        skippable: bool = False,
        checkpoint_metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize WaitingForUserInputException.

        Args:
            question: Question to ask the user
            input_type: Type of input - "text", "approval", "select", or "multiselect"
            options: List of option dicts (for approval/select/multiselect)
            checkpoint_state: Workflow state snapshot for resume
            pause_index: Index of this pause point (0-indexed, for multi-step HITL)
            agent_context: Optional agent execution state for resuming agents
                Required fields when provided:
                - agent_name: Name of the agent that paused
                - iteration: Current iteration number (0-indexed)
                - messages: LLM conversation history as list of dicts
                - tool_results: Partial tool results for current iteration
                - pending_tool_call: The HITL tool call awaiting response
                - all_tool_calls: All tool calls made so far
                - model_config: Model settings for resume
            step_name: Name of the wait_for_user step for event correlation
            step_correlation_id: Correlation ID for the step events
            allow_custom: Whether to allow a free-text "Something else" option
            skippable: Whether the user can skip this input
            checkpoint_metadata: Complete metadata needed to resume this pause
        """
        super().__init__(f"Waiting for user input: {question}")
        self.question = question
        self.input_type = input_type
        self.options = options or []
        self.checkpoint_state = checkpoint_state
        self.pause_index = pause_index
        self.agent_context = agent_context
        self.step_name = step_name
        self.step_correlation_id = step_correlation_id
        self.allow_custom = allow_custom
        self.skippable = skippable
        self.checkpoint_metadata = checkpoint_metadata or {}
