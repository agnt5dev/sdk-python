"""AGNT5 SDK exceptions and error types."""


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
