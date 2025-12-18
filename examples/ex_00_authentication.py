"""
AGNT5 Authentication Examples

This example demonstrates the two authentication methods for accessing
AGNT5 deployments:

1. Subdomain URLs - Recommended for production apps
2. API Keys (Service Keys) - For programmatic access

Run with:
    python ex_00_authentication.py

Environment Variables:
    AGNT5_API_KEY - Service key for API key authentication
"""

import os
from agnt5 import Client


def example_subdomain_auth():
    """
    Authentication via subdomain URL.

    This is the recommended approach for production applications.
    The project and environment are encoded in the URL.

    URL Format:
    - Production: {project}-prod.run.agnt5.com
    - Staging:    {project}-stg.run.agnt5.com
    - Preview:    {project}--{deployment-ref}.run.agnt5.com
    """
    print("\n=== Subdomain Authentication ===\n")

    # Production environment
    client = Client("https://myproject-prod.run.agnt5.com")
    print(f"Production client: {client.gateway_url}")

    # Staging environment
    staging_client = Client("https://myproject-stg.run.agnt5.com")
    print(f"Staging client: {staging_client.gateway_url}")

    # Preview deployment (note the double dash)
    preview_client = Client("https://myproject--abc123ef.run.agnt5.com")
    print(f"Preview client: {preview_client.gateway_url}")

    # Example: Run a workflow
    # result = client.run("my_workflow", {"input": "data"})
    # print(f"Result: {result}")


def example_api_key_auth():
    """
    Authentication via API key (service key).

    Use this approach for:
    - CI/CD pipelines
    - Serverless functions
    - Scripts and automation
    - Multi-environment access with a single key

    The API key can be passed directly or via environment variable.
    """
    print("\n=== API Key Authentication ===\n")

    # Option 1: Pass api_key directly
    # (Don't hardcode in production - use env vars or secret manager)
    client = Client(
        gateway_url="https://api.agnt5.com",
        api_key="agnt5_sk_example_key_not_real"
    )
    print(f"Direct API key: {client.api_key[:20]}..." if client.api_key else "No key")

    # Option 2: Use AGNT5_API_KEY environment variable
    # Set the env var first:
    #   export AGNT5_API_KEY=agnt5_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    os.environ["AGNT5_API_KEY"] = "agnt5_sk_from_environment"

    env_client = Client(gateway_url="https://api.agnt5.com")
    print(f"Env var API key: {env_client.api_key[:20]}..." if env_client.api_key else "No key")

    # Clean up example env var
    del os.environ["AGNT5_API_KEY"]

    # Example: Run a workflow with API key auth
    # result = client.run("my_workflow", {"input": "data"})
    # print(f"Result: {result}")


def example_local_development():
    """
    Local development authentication.

    During local development, you can use either method.
    The dev server accepts both authenticated and unauthenticated requests.
    """
    print("\n=== Local Development ===\n")

    # Default local development URL (no auth required)
    local_client = Client("http://localhost:34181")
    print(f"Local client: {local_client.gateway_url}")

    # Example: Run a workflow locally
    # result = local_client.run("greet", {"name": "World"})
    # print(f"Result: {result}")


def example_async_client():
    """
    Using the async client with authentication.

    The AsyncClient supports the same authentication methods.
    """
    print("\n=== Async Client Authentication ===\n")

    from agnt5 import AsyncClient

    # With subdomain URL
    async_client = AsyncClient("https://myproject-prod.run.agnt5.com")
    print(f"Async subdomain client: {async_client.gateway_url}")

    # With API key
    async_client_with_key = AsyncClient(
        gateway_url="https://api.agnt5.com",
        api_key="agnt5_sk_example_key_not_real"
    )
    print(f"Async API key client: {async_client_with_key.gateway_url}")

    # Example: Run a workflow asynchronously
    # async def run_workflow():
    #     result = await async_client.run("my_workflow", {"input": "data"})
    #     print(f"Result: {result}")
    # import asyncio
    # asyncio.run(run_workflow())


def example_best_practices():
    """
    Best practices for authentication.
    """
    print("\n=== Best Practices ===\n")

    print("""
    1. Never commit API keys to version control
       - Add .env to .gitignore
       - Use secret managers in production

    2. Use environment variables
       - export AGNT5_API_KEY=agnt5_sk_xxx
       - Use python-dotenv for .env files

    3. Use minimum required scopes
       - Request only the scopes you need
       - Separate keys for different services

    4. Rotate keys regularly
       - Set calendar reminders
       - Use key expiration for temporary access

    5. Use subdomain URLs when possible
       - Simpler, no key management
       - Environment isolation built-in

    6. Implement backend proxies for client apps
       - Never expose keys to frontend code
       - Your backend proxies AGNT5 requests
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("AGNT5 Authentication Examples")
    print("=" * 60)

    example_subdomain_auth()
    example_api_key_auth()
    example_local_development()
    example_async_client()
    example_best_practices()

    print("\n" + "=" * 60)
    print("See docs/guide/authentication.md for more details")
    print("=" * 60)
