"""Version information for agnt5 SDK."""


def _get_version() -> str:
    """Get package version from installed metadata.

    This uses importlib.metadata (Python 3.8+) to read the version from
    the installed package metadata, maintaining pyproject.toml as the
    single source of truth.

    Returns:
        Package version string, or "0.0.0+dev" for development installs.
    """
    try:
        from importlib.metadata import version
        return version("agnt5")
    except Exception:
        # Development/editable install fallback
        return "0.0.0+dev"
