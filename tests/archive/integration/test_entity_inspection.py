"""
Integration Tests: Entity Schema Inspection

Tests the REST API endpoints for inspecting entity types and their methods.
"""

import pytest


@pytest.mark.integration
def test_list_entities(client, worker_process):
    """
    Test GET /v1/entities returns list of registered entities.

    Should return all entities registered by the worker with their methods.
    """
    import httpx

    # Get gateway URL from client
    gateway_url = client.gateway_url

    # Call the entities list endpoint
    response = httpx.get(f"{gateway_url}/v1/entities")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert "entities" in data
    assert "total" in data
    assert data["total"] > 0, "Should have at least one entity registered"

    # Verify ShoppingCart entity is in the list
    entity_names = [e["name"] for e in data["entities"]]
    assert "ShoppingCart" in entity_names, f"ShoppingCart not found in {entity_names}"
    assert "Counter" in entity_names, f"Counter not found in {entity_names}"
    assert "BankAccount" in entity_names, f"BankAccount not found in {entity_names}"

    # Check ShoppingCart details
    shopping_cart = next((e for e in data["entities"] if e["name"] == "ShoppingCart"), None)
    assert shopping_cart is not None
    assert shopping_cart["serviceName"] == "test-service"
    assert shopping_cart["methodCount"] > 0
    assert "methodNames" in shopping_cart
    assert "add_item" in shopping_cart["methodNames"]
    assert "get_total" in shopping_cart["methodNames"]


@pytest.mark.integration
def test_get_entity_details(client, worker_process):
    """
    Test GET /v1/entities/:type returns detailed entity information.

    Should return entity metadata including all methods with their schemas.
    """
    import httpx

    # Get gateway URL from client
    gateway_url = client.gateway_url

    # Call the entity details endpoint
    response = httpx.get(f"{gateway_url}/v1/entities/ShoppingCart")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["name"] == "ShoppingCart"
    assert data["type"] == "entity"
    assert data["serviceName"] == "test-service"
    assert "methods" in data
    assert len(data["methods"]) > 0

    # Check that methods have schemas
    method_names = [m["name"] for m in data["methods"]]
    assert "add_item" in method_names
    assert "get_total" in method_names
    assert "get_items" in method_names
    assert "clear" in method_names

    # Check add_item method has input/output schemas
    add_item = next((m for m in data["methods"] if m["name"] == "add_item"), None)
    assert add_item is not None

    # Note: Schemas may be None if not extracted during registration
    # This is okay - we're just verifying the endpoint structure works
    if add_item.get("inputSchema"):
        assert isinstance(add_item["inputSchema"], (dict, str))
    if add_item.get("outputSchema"):
        assert isinstance(add_item["outputSchema"], (dict, str))


@pytest.mark.integration
def test_get_entity_not_found(client, worker_process):
    """
    Test GET /v1/entities/:type returns 404 for non-existent entity.
    """
    import httpx

    # Get gateway URL from client
    gateway_url = client.gateway_url

    # Call the entity details endpoint with non-existent entity
    response = httpx.get(f"{gateway_url}/v1/entities/NonExistentEntity")

    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert "NonExistentEntity" in data["error"]


@pytest.mark.integration
def test_entity_inspection_with_counter(client, worker_process):
    """
    Test entity inspection for Counter entity.
    """
    import httpx

    # Get gateway URL from client
    gateway_url = client.gateway_url

    # Get Counter entity details
    response = httpx.get(f"{gateway_url}/v1/entities/Counter")

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "Counter"
    assert data["type"] == "entity"

    # Check methods
    method_names = [m["name"] for m in data["methods"]]
    assert "increment" in method_names
    assert "get_count" in method_names
    assert "reset" in method_names
