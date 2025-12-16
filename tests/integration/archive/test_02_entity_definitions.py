"""
Integration tests for entity definition persistence.

Verifies that entity definitions (method schemas and state schemas) are properly
stored in the platform database and accessible via MCP tools.
"""

import json
import sqlite3
import pytest


@pytest.mark.integration
def test_entity_definition_persisted(worker_process, platform):
    """Verify entity definitions are stored in component_definition field."""
    # Get database path
    db_path = platform['host_db_path']

    # Wait for worker registration to complete
    import time
    time.sleep(2)

    # Query database for entity component definitions
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT
            component_name,
            component_type,
            component_definition
        FROM components
        WHERE component_type = 'COMPONENT_TYPE_ENTITY'
        AND component_definition IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    # Should have at least 3 entities (ShoppingCart, Counter, BankAccount)
    assert len(rows) >= 3, f"Expected at least 3 entities with definitions, got {len(rows)}"

    # Verify each entity has a valid definition
    entity_definitions = {}
    for name, component_type, definition_str in rows:
        assert definition_str is not None, f"Entity {name} has NULL definition"
        assert len(definition_str) > 0, f"Entity {name} has empty definition"

        # Parse definition JSON
        definition = json.loads(definition_str)
        entity_definitions[name] = definition

        # Verify definition structure
        assert "entity_name" in definition, f"Entity {name} missing entity_name in definition"
        assert "methods" in definition, f"Entity {name} missing methods in definition"
        assert isinstance(definition["methods"], dict), f"Entity {name} methods is not a dict"

        # Optional state_schema (at least one entity should have it)
        if "state_schema" in definition:
            assert isinstance(definition["state_schema"], dict), f"Entity {name} state_schema is not a dict"
            assert "type" in definition["state_schema"], f"Entity {name} state_schema missing type"

    # Verify specific entities
    entity_names = [name for name, _, _ in rows]
    assert "ShoppingCart" in entity_names, "ShoppingCart entity not found"
    assert "Counter" in entity_names, "Counter entity not found"
    assert "BankAccount" in entity_names, "BankAccount entity not found"


@pytest.mark.integration
def test_entity_method_schemas_captured(worker_process, platform):
    """Verify entity method schemas are captured correctly."""
    db_path = platform['host_db_path']

    # Wait for registration
    import time
    time.sleep(2)

    # Query ShoppingCart entity
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT component_definition
        FROM components
        WHERE component_type = 'COMPONENT_TYPE_ENTITY'
        AND component_name = 'ShoppingCart'
    """)

    row = cursor.fetchone()
    conn.close()

    assert row is not None, "ShoppingCart entity not found"

    definition = json.loads(row[0])
    methods = definition.get("methods", {})

    # Verify expected methods exist
    expected_methods = ["add_item", "get_total", "get_items", "clear"]
    for method_name in expected_methods:
        assert method_name in methods, f"Method {method_name} not found in ShoppingCart"

        method_def = methods[method_name]
        # Each method should have input_schema, output_schema
        assert "input_schema" in method_def or "output_schema" in method_def, \
            f"Method {method_name} missing schemas"


@pytest.mark.integration
def test_entity_state_schemas_captured(worker_process, platform):
    """Verify entity state schemas are captured correctly."""
    db_path = platform['host_db_path']

    # Wait for registration
    import time
    time.sleep(2)

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT component_name, component_definition
        FROM components
        WHERE component_type = 'COMPONENT_TYPE_ENTITY'
        AND component_definition IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    # Count entities with state schemas
    entities_with_state_schema = []
    for name, definition_str in rows:
        definition = json.loads(definition_str)
        if "state_schema" in definition:
            entities_with_state_schema.append(name)

            state_schema = definition["state_schema"]
            # Verify state schema has required JSON Schema fields
            assert "type" in state_schema, f"Entity {name} state_schema missing 'type'"
            assert state_schema["type"] == "object", f"Entity {name} state_schema type is not 'object'"

            # Should have properties or description
            assert "properties" in state_schema or "description" in state_schema, \
                f"Entity {name} state_schema should have 'properties' or 'description'"

    # All three entities should have explicit state schemas
    assert len(entities_with_state_schema) >= 3, \
        f"Expected at least 3 entities with state schemas, found {len(entities_with_state_schema)}: {entities_with_state_schema}"


@pytest.mark.integration
def test_entity_definition_structure(worker_process, platform):
    """Verify entity definition follows expected structure."""
    db_path = platform['host_db_path']

    # Wait for registration
    import time
    time.sleep(2)

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT component_name, component_definition
        FROM components
        WHERE component_type = 'COMPONENT_TYPE_ENTITY'
        AND component_name = 'BankAccount'
    """)

    row = cursor.fetchone()
    conn.close()

    assert row is not None, "BankAccount entity not found"

    name, definition_str = row
    definition = json.loads(definition_str)

    # Verify complete structure
    assert definition["entity_name"] == "BankAccount"

    # Verify methods
    assert "methods" in definition
    methods = definition["methods"]
    assert "deposit" in methods
    assert "withdraw" in methods
    assert "get_balance" in methods
    assert "get_transactions" in methods

    # Verify state schema
    assert "state_schema" in definition
    state_schema = definition["state_schema"]
    assert "properties" in state_schema

    # BankAccount should have balance and transactions properties
    properties = state_schema["properties"]
    assert "balance" in properties
    assert "transactions" in properties

    # Verify property types
    assert properties["balance"]["type"] == "number"
    assert properties["transactions"]["type"] == "array"


@pytest.mark.integration
def test_counter_state_schema_details(worker_process, platform):
    """Verify Counter entity has detailed state schema."""
    db_path = platform['host_db_path']

    # Wait for registration
    import time
    time.sleep(2)

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT component_definition
        FROM components
        WHERE component_type = 'COMPONENT_TYPE_ENTITY'
        AND component_name = 'Counter'
    """)

    row = cursor.fetchone()
    conn.close()

    assert row is not None, "Counter entity not found"

    definition = json.loads(row[0])

    # Verify state schema
    assert "state_schema" in definition
    state_schema = definition["state_schema"]

    # Counter should have count property
    assert "properties" in state_schema
    properties = state_schema["properties"]
    assert "count" in properties

    # count should be integer type
    count_schema = properties["count"]
    assert count_schema["type"] == "integer"
    assert "description" in count_schema
