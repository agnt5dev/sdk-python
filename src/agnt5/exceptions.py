"""AGNT5 SDK exceptions and error types."""

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
    """Raised when a function exceeds maximum retry attempts."""

    def __init__(self, message: str, attempts: int, last_error: Exception) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class StateError(AGNT5Error):
    """Raised when state operations fail."""

    pass


class CheckpointError(AGNT5Error):
    """Raised when checkpoint operations fail."""

    pass


class NotImplementedError(AGNT5Error):
    """Raised when a feature is not yet implemented."""

    pass


class WaitingForUserInputException(AGNT5Error):
    """Raised when workflow needs to pause for user input.

    This exception is used internally by ctx.wait_for_user() to signal
    that a workflow execution should pause and wait for user input.

    The platform catches this exception and:
    1. Saves the workflow checkpoint state
    2. Returns awaiting_user_input status to the client
    3. Presents the question and options to the user
    4. Resumes execution when user responds

    Attributes:
        question: The question to ask the user
        input_type: Type of input ("text", "approval", or "choice")
        options: List of options for approval/choice inputs
        checkpoint_state: Current workflow state for resume
        agent_context: Optional agent execution state for agent-level HITL
            Contains: agent_name, iteration, messages, tool_results, pending_tool_call, etc.
    """

    def __init__(
        self,
        question: str,
        input_type: str,
        options: Optional[List[Dict]],
        checkpoint_state: Dict,
        agent_context: Optional[Dict] = None,
    ) -> None:
        """Initialize WaitingForUserInputException.

        Args:
            question: Question to ask the user
            input_type: Type of input - "text", "approval", or "choice"
            options: List of option dicts (for approval/choice)
            checkpoint_state: Workflow state snapshot for resume
            agent_context: Optional agent execution state for resuming agents
                Required fields when provided:
                - agent_name: Name of the agent that paused
                - iteration: Current iteration number (0-indexed)
                - messages: LLM conversation history as list of dicts
                - tool_results: Partial tool results for current iteration
                - pending_tool_call: The HITL tool call awaiting response
                - all_tool_calls: All tool calls made so far
                - model_config: Model settings for resume
        """
        super().__init__(f"Waiting for user input: {question}")
        self.question = question
        self.input_type = input_type
        self.options = options or []
        self.checkpoint_state = checkpoint_state
        self.agent_context = agent_context
