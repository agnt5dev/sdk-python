"""
Example: Graph Memory API - Knowledge Relationships

Demonstrates knowledge graph operations with the Memory API:
- Creating nodes and relationships
- Querying relationships
- Graph traversal with filters
- Building knowledge graphs for RAG and agents

This example shows how to:
1. Create a knowledge graph of users, topics, and documents
2. Query relationships between entities
3. Traverse graphs to discover connections
4. Use graphs for context-aware AI applications
"""

import asyncio
from agnt5 import (
    GraphMemory,
    GraphNode,
    GraphRelationship,
    GraphTraversalResult,
)


async def demo_basic_graph_operations():
    """Demonstrate basic graph operations."""
    print("=" * 60)
    print("Basic Graph Operations")
    print("=" * 60)

    # Create graph scoped to a user
    graph = GraphMemory(scope="user:alice")

    print("\n1. Creating nodes...")

    # Create user nodes
    await graph.upsert_node("User:alice", "User", {"name": "Alice"})
    await graph.upsert_node("User:bob", "User", {"name": "Bob"})

    # Create topic nodes
    await graph.upsert_node("Topic:python", "Topic", {"name": "Python"})
    await graph.upsert_node("Topic:ai", "Topic", {"name": "Artificial Intelligence"})
    await graph.upsert_node("Topic:ml", "Topic", {"name": "Machine Learning"})

    # Create document nodes
    await graph.upsert_node("Doc:guide", "Document", {"title": "Python AI Guide"})

    print("   ✓ Created 6 nodes (2 users, 3 topics, 1 document)")

    print("\n2. Creating relationships...")

    # User interests
    rel1 = await graph.relate("User:alice", "Topic:python", "interested_in")
    rel2 = await graph.relate("User:alice", "Topic:ai", "interested_in")
    rel3 = await graph.relate("User:bob", "Topic:ml", "interested_in")

    # User connections
    rel4 = await graph.relate("User:alice", "User:bob", "knows")

    # Topic relationships
    rel5 = await graph.relate("Topic:ml", "Topic:ai", "subtopic_of")
    rel6 = await graph.relate("Topic:python", "Topic:ml", "used_for")

    # Document relationships
    rel7 = await graph.relate("Doc:guide", "Topic:python", "about")
    rel8 = await graph.relate("Doc:guide", "Topic:ai", "about")

    print(f"   ✓ Created 8 relationships")

    print("\n3. Querying relationships...")

    # Get Alice's interests
    alice_interests = await graph.query_relationships(
        from_node="User:alice",
        relationship_type="interested_in"
    )

    print(f"\n   Alice is interested in:")
    for rel in alice_interests:
        print(f"   - {rel.to_node}")

    # Get who knows Bob
    bob_connections = await graph.query_relationships(
        to_node="User:bob",
        relationship_type="knows"
    )

    print(f"\n   People who know Bob:")
    for rel in bob_connections:
        print(f"   - {rel.from_node}")

    print()


async def demo_graph_traversal():
    """Demonstrate graph traversal to discover connections."""
    print("=" * 60)
    print("Graph Traversal - Discovering Connections")
    print("=" * 60)

    # Create a knowledge graph
    graph = GraphMemory(scope="user:charlie")

    # Create nodes
    await graph.upsert_node("User:charlie", "User")
    await graph.upsert_node("Topic:databases", "Topic")
    await graph.upsert_node("Topic:sql", "Topic")
    await graph.upsert_node("Topic:nosql", "Topic")
    await graph.upsert_node("Tool:postgres", "Tool")
    await graph.upsert_node("Tool:mongodb", "Tool")
    await graph.upsert_node("Doc:sql_guide", "Document")

    # Create relationships
    await graph.relate("User:charlie", "Topic:databases", "learning")
    await graph.relate("Topic:databases", "Topic:sql", "subtopic")
    await graph.relate("Topic:databases", "Topic:nosql", "subtopic")
    await graph.relate("Topic:sql", "Tool:postgres", "implemented_by")
    await graph.relate("Topic:nosql", "Tool:mongodb", "implemented_by")
    await graph.relate("Doc:sql_guide", "Topic:sql", "teaches")

    print("\n1. Traverse from user (depth=2)...")

    # Traverse 2 levels deep from user
    result = await graph.traverse("User:charlie", depth=2)

    print(f"\n   Found {len(result.nodes)} nodes:")
    for node in result.nodes:
        print(f"   - {node.node_type}: {node.id}")

    print(f"\n   Found {len(result.relationships)} relationships:")
    for rel in result.relationships:
        print(f"   - {rel.from_node} -[{rel.relationship_type}]-> {rel.to_node}")

    print("\n2. Traverse with filters (only subtopic relationships)...")

    # Traverse filtering by relationship type
    result = await graph.traverse(
        "Topic:databases",
        depth=2,
        relationship_types=["subtopic", "implemented_by"]
    )

    print(f"\n   Found {len(result.nodes)} nodes (filtered):")
    for node in result.nodes:
        print(f"   - {node.node_type}: {node.id}")

    print()


async def demo_knowledge_graph_for_rag():
    """Demonstrate building a knowledge graph for RAG applications."""
    print("=" * 60)
    print("Knowledge Graph for RAG - Context Discovery")
    print("=" * 60)

    # Create a knowledge graph for documentation
    graph = GraphMemory(scope="docs:agnt5")

    print("\n1. Building documentation knowledge graph...")

    # Create concept nodes
    concepts = [
        ("Concept:workflow", "Concept", "Workflow"),
        ("Concept:function", "Concept", "Function"),
        ("Concept:entity", "Concept", "Entity"),
        ("Concept:state", "Concept", "State"),
        ("Concept:memory", "Concept", "Memory"),
    ]

    for node_id, node_type, name in concepts:
        await graph.upsert_node(node_id, node_type, {"name": name})

    # Create document nodes
    docs = [
        ("Doc:workflow_guide", "Document", "Workflow Guide"),
        ("Doc:function_guide", "Document", "Function Guide"),
        ("Doc:state_guide", "Document", "State Guide"),
    ]

    for node_id, node_type, title in docs:
        await graph.upsert_node(node_id, node_type, {"title": title})

    # Create relationships between concepts
    await graph.relate("Concept:workflow", "Concept:function", "uses")
    await graph.relate("Concept:workflow", "Concept:entity", "uses")
    await graph.relate("Concept:entity", "Concept:state", "has")
    await graph.relate("Concept:function", "Concept:state", "accesses")

    # Link documents to concepts
    await graph.relate("Doc:workflow_guide", "Concept:workflow", "explains")
    await graph.relate("Doc:workflow_guide", "Concept:function", "mentions")
    await graph.relate("Doc:function_guide", "Concept:function", "explains")
    await graph.relate("Doc:state_guide", "Concept:state", "explains")
    await graph.relate("Doc:state_guide", "Concept:memory", "explains")

    print("   ✓ Created 5 concepts, 3 documents, 9 relationships")

    print("\n2. Query: What docs are related to workflows?")

    # Find all documents connected to workflow concept
    result = await graph.traverse(
        "Concept:workflow",
        depth=2,
        node_types=["Document"]  # Only return documents
    )

    print(f"\n   Found {len(result.nodes)} related documents:")
    for node in result.nodes:
        if node.node_type == "Document":
            print(f"   - {node.id}")

    print("\n3. Query: What concepts are related to state?")

    # Find concepts related to state
    result = await graph.traverse(
        "Concept:state",
        depth=2,
        node_types=["Concept"]
    )

    print(f"\n   Found {len(result.nodes)} related concepts:")
    for node in result.nodes:
        if node.id != "Concept:state":  # Exclude the starting node
            print(f"   - {node.id}")

    print()


async def demo_user_preference_graph():
    """Demonstrate building a user preference graph."""
    print("=" * 60)
    print("User Preference Graph - Personalization")
    print("=" * 60)

    # Create user preference graph
    graph = GraphMemory(scope="user:dana")

    print("\n1. Building user preference graph...")

    # Create user and preference nodes
    await graph.upsert_node("User:dana", "User", {"name": "Dana"})
    await graph.upsert_node("Pref:dark_mode", "Preference", {"setting": "theme"})
    await graph.upsert_node("Pref:concise", "Preference", {"setting": "verbosity"})
    await graph.upsert_node("Lang:python", "Language", {"name": "Python"})
    await graph.upsert_node("Lang:typescript", "Language", {"name": "TypeScript"})
    await graph.upsert_node("Framework:react", "Framework", {"name": "React"})

    # Create relationships
    await graph.relate("User:dana", "Pref:dark_mode", "prefers")
    await graph.relate("User:dana", "Pref:concise", "prefers")
    await graph.relate("User:dana", "Lang:python", "uses", {"skill": "expert"})
    await graph.relate("User:dana", "Lang:typescript", "uses", {"skill": "intermediate"})
    await graph.relate("User:dana", "Framework:react", "uses", {"skill": "advanced"})

    print("   ✓ Created user preference graph")

    print("\n2. Query: What does Dana prefer?")

    prefs = await graph.query_relationships(
        from_node="User:dana",
        relationship_type="prefers"
    )

    print(f"\n   Dana's preferences:")
    for rel in prefs:
        print(f"   - {rel.to_node}")

    print("\n3. Query: What languages does Dana use?")

    langs = await graph.query_relationships(
        from_node="User:dana",
        relationship_type="uses"
    )

    print(f"\n   Dana uses:")
    for rel in langs:
        skill = rel.properties.get("skill", "unknown")
        print(f"   - {rel.to_node} (skill: {skill})")

    print("\n4. Full context (traverse from user)...")

    # Get full user context
    result = await graph.traverse("User:dana", depth=1)

    print(f"\n   Complete user profile has:")
    print(f"   - {len(result.nodes)} nodes (including user)")
    print(f"   - {len(result.relationships)} relationships")

    print()


async def main():
    """Run all graph memory examples."""
    print("\n")
    print("=" * 60)
    print("AGNT5 Graph Memory Examples")
    print("=" * 60)
    print()

    # Run demos
    await demo_basic_graph_operations()
    await demo_graph_traversal()
    await demo_knowledge_graph_for_rag()
    await demo_user_preference_graph()

    print("=" * 60)
    print("Graph Memory Demo Complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("- GraphMemory provides knowledge graph operations")
    print("- Create nodes and relationships to model domain knowledge")
    print("- Query and traverse to discover connections")
    print("- Use for RAG, personalization, and context-aware AI")
    print()


if __name__ == "__main__":
    asyncio.run(main())
