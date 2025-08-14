from ._compat import _rust_available, _import_error

def get_worker(
    service_name: str,
    service_version: str = "1.0.0",
    coordinator_endpoint: str = None,
    auto_register: bool = True,
) -> "DurableWorker":
    """
    Create a new durable worker using the Rust core.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        coordinator_endpoint: Endpoint of the coordinator service
        auto_register: Whether to automatically register decorated components

    Returns:
        A configured DurableWorker instance

    Raises:
        RuntimeError: If the Rust core is not available
    """
    if not _rust_available:
        raise RuntimeError(f"Rust core is required but not available: {_import_error}. Please build and install the Rust extension first.")

    # Use the high-performance Rust core
    import uuid

    worker_id = str(uuid.uuid4())

    rust_worker = create_worker(
        worker_id=worker_id,
        service_name=service_name,
        version=service_version,
        coordinator_endpoint=coordinator_endpoint,
    )

    worker = DurableWorker(rust_worker)
    
    # Store auto-registration preference for later use
    worker._auto_register = auto_register
    
    return worker
