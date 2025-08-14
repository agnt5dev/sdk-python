

# Read version from pyproject.toml to maintain single source of truth
def _get_version():
    try:
        import tomllib
    except ImportError:
        # Python < 3.11 fallback
        try:
            import tomli as tomllib
        except ImportError:
            # Final fallback if no toml library available
            return "UNKNOWN"
    
    try:
        import pathlib
        pyproject_path = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        return pyproject_data["project"]["version"]
    except Exception:
        # Fallback version if reading fails
        return "UNKNOWN"