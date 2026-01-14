/// PyO3 bindings for GraphDatabase
///
/// Provides Python async access to the Rust graph database implementation.

use agnt5_sdk_core::graph::{
    GraphDatabase, GraphNode, GraphRelationship, GraphTraversalResult,
    RelationshipQuery, TraversalFilters,
};
use agnt5_sdk_core::MemoryGraphDatabase;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Python-accessible graph node
#[pyclass(name = "GraphNode")]
#[derive(Clone)]
pub struct PyGraphNode {
    /// Node ID
    #[pyo3(get)]
    pub id: String,

    /// Node type
    #[pyo3(get)]
    pub node_type: String,

    /// Node properties
    #[pyo3(get)]
    pub properties: HashMap<String, String>,
}

#[pymethods]
impl PyGraphNode {
    fn __repr__(&self) -> String {
        format!("GraphNode(id='{}', type='{}')", self.id, self.node_type)
    }

    fn __str__(&self) -> String {
        format!("{}[{}]", self.node_type, self.id)
    }
}

impl From<GraphNode> for PyGraphNode {
    fn from(n: GraphNode) -> Self {
        // Convert JSON properties to string map
        let properties = n
            .properties
            .into_iter()
            .filter_map(|(k, v)| match v {
                Value::String(s) => Some((k, s)),
                Value::Number(num) => Some((k, num.to_string())),
                Value::Bool(b) => Some((k, b.to_string())),
                _ => None,
            })
            .collect();

        Self {
            id: n.id,
            node_type: n.node_type,
            properties,
        }
    }
}

/// Python-accessible graph relationship
#[pyclass(name = "GraphRelationship")]
#[derive(Clone)]
pub struct PyGraphRelationship {
    /// Relationship ID
    #[pyo3(get)]
    pub id: String,

    /// Source node ID
    #[pyo3(get)]
    pub from_node: String,

    /// Target node ID
    #[pyo3(get)]
    pub to_node: String,

    /// Relationship type
    #[pyo3(get)]
    pub relationship_type: String,

    /// Relationship properties
    #[pyo3(get)]
    pub properties: HashMap<String, String>,

    /// Creation timestamp
    #[pyo3(get)]
    pub created_at: i64,
}

#[pymethods]
impl PyGraphRelationship {
    fn __repr__(&self) -> String {
        format!(
            "GraphRelationship(from='{}', type='{}', to='{}')",
            self.from_node, self.relationship_type, self.to_node
        )
    }

    fn __str__(&self) -> String {
        format!(
            "{} -[{}]-> {}",
            self.from_node, self.relationship_type, self.to_node
        )
    }
}

impl From<GraphRelationship> for PyGraphRelationship {
    fn from(r: GraphRelationship) -> Self {
        // Convert JSON properties to string map
        let properties = r
            .properties
            .into_iter()
            .filter_map(|(k, v)| match v {
                Value::String(s) => Some((k, s)),
                Value::Number(num) => Some((k, num.to_string())),
                Value::Bool(b) => Some((k, b.to_string())),
                _ => None,
            })
            .collect();

        Self {
            id: r.id,
            from_node: r.from_node,
            to_node: r.to_node,
            relationship_type: r.relationship_type,
            properties,
            created_at: r.created_at,
        }
    }
}

/// Python-accessible graph traversal result
#[pyclass(name = "GraphTraversalResult")]
#[derive(Clone)]
pub struct PyGraphTraversalResult {
    /// Starting node
    #[pyo3(get)]
    pub start_node: String,

    /// All nodes found
    #[pyo3(get)]
    pub nodes: Vec<PyGraphNode>,

    /// All relationships found
    #[pyo3(get)]
    pub relationships: Vec<PyGraphRelationship>,

    /// Traversal depth
    #[pyo3(get)]
    pub depth: u32,
}

#[pymethods]
impl PyGraphTraversalResult {
    fn __repr__(&self) -> String {
        format!(
            "GraphTraversalResult(start='{}', nodes={}, relationships={})",
            self.start_node,
            self.nodes.len(),
            self.relationships.len()
        )
    }
}

impl From<GraphTraversalResult> for PyGraphTraversalResult {
    fn from(r: GraphTraversalResult) -> Self {
        Self {
            start_node: r.start_node,
            nodes: r.nodes.into_iter().map(PyGraphNode::from).collect(),
            relationships: r
                .relationships
                .into_iter()
                .map(PyGraphRelationship::from)
                .collect(),
            depth: r.depth,
        }
    }
}

/// Python-accessible in-memory graph database
#[pyclass(name = "MemoryGraph")]
pub struct PyMemoryGraph {
    inner: Arc<MemoryGraphDatabase>,
}

#[pymethods]
impl PyMemoryGraph {
    /// Create a new in-memory graph database
    ///
    /// Args:
    ///     scope: Scope identifier for isolation (e.g., "user:123")
    #[new]
    fn new(scope: String) -> Self {
        Self {
            inner: Arc::new(MemoryGraphDatabase::new(scope)),
        }
    }

    /// Create or update a node
    ///
    /// Args:
    ///     node_id: Unique node identifier
    ///     node_type: Type of node (e.g., "User", "Topic")
    ///     properties: Optional node properties (dict)
    ///
    /// Returns:
    ///     The node ID
    fn upsert_node<'a>(
        &self,
        py: Python<'a>,
        node_id: String,
        node_type: String,
        properties: Option<HashMap<String, String>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let graph = self.inner.clone();
        let props_map: HashMap<String, Value> = properties
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, Value::String(v)))
            .collect();

        future_into_py(py, async move {
            graph
                .upsert_node(&node_id, &node_type, props_map)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to upsert node: {}",
                        e
                    ))
                })
        })
    }

    /// Create a relationship between two nodes
    ///
    /// Args:
    ///     from_node: Source node ID
    ///     to_node: Target node ID
    ///     relationship_type: Type of relationship
    ///     properties: Optional relationship properties (dict)
    ///
    /// Returns:
    ///     Unique relationship ID
    fn create_relationship<'a>(
        &self,
        py: Python<'a>,
        from_node: String,
        to_node: String,
        relationship_type: String,
        properties: Option<HashMap<String, String>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let graph = self.inner.clone();
        let props_map: HashMap<String, Value> = properties
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, Value::String(v)))
            .collect();

        future_into_py(py, async move {
            graph
                .create_relationship(&from_node, &to_node, &relationship_type, props_map)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create relationship: {}",
                        e
                    ))
                })
        })
    }

    /// Query relationships matching criteria
    ///
    /// Args:
    ///     from_node: Filter by source node (optional)
    ///     to_node: Filter by target node (optional)
    ///     relationship_type: Filter by relationship type (optional)
    ///     limit: Maximum results (default: 100)
    ///
    /// Returns:
    ///     List of matching relationships
    fn query_relationships<'a>(
        &self,
        py: Python<'a>,
        from_node: Option<String>,
        to_node: Option<String>,
        relationship_type: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let graph = self.inner.clone();
        let query = RelationshipQuery {
            from_node,
            to_node,
            relationship_type,
            limit: limit.unwrap_or(100),
        };

        future_into_py(py, async move {
            graph
                .query_relationships(query)
                .await
                .map(|rels| {
                    rels.into_iter()
                        .map(PyGraphRelationship::from)
                        .collect::<Vec<_>>()
                })
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to query relationships: {}",
                        e
                    ))
                })
        })
    }

    /// Traverse graph from a starting node
    ///
    /// Args:
    ///     start_node: Starting node ID
    ///     max_depth: Maximum depth to traverse
    ///     relationship_types: Filter by relationship types (optional)
    ///     node_types: Filter by node types (optional)
    ///
    /// Returns:
    ///     Graph traversal result with nodes and relationships
    fn traverse<'a>(
        &self,
        py: Python<'a>,
        start_node: String,
        max_depth: u32,
        relationship_types: Option<Vec<String>>,
        node_types: Option<Vec<String>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let graph = self.inner.clone();
        let filters = if relationship_types.is_some() || node_types.is_some() {
            Some(TraversalFilters {
                relationship_types,
                node_types,
            })
        } else {
            None
        };

        future_into_py(py, async move {
            graph
                .traverse(&start_node, max_depth, filters)
                .await
                .map(PyGraphTraversalResult::from)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to traverse graph: {}",
                        e
                    ))
                })
        })
    }

    /// Get a node by its ID
    ///
    /// Args:
    ///     node_id: Node identifier
    ///
    /// Returns:
    ///     Node if found, None otherwise
    fn get_node<'a>(&self, py: Python<'a>, node_id: String) -> PyResult<Bound<'a, PyAny>> {
        let graph = self.inner.clone();

        future_into_py(py, async move {
            graph
                .get_node(&node_id)
                .await
                .map(|opt| opt.map(PyGraphNode::from))
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get node: {}", e))
                })
        })
    }

    /// Delete a relationship by its ID
    ///
    /// Args:
    ///     relationship_id: Relationship identifier
    ///
    /// Returns:
    ///     True if deleted, False if not found
    fn delete_relationship<'a>(
        &self,
        py: Python<'a>,
        relationship_id: String,
    ) -> PyResult<Bound<'a, PyAny>> {
        let graph = self.inner.clone();

        future_into_py(py, async move {
            graph.delete_relationship(&relationship_id).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to delete relationship: {}",
                    e
                ))
            })
        })
    }

    /// Delete a node and all its relationships
    ///
    /// Args:
    ///     node_id: Node identifier
    ///
    /// Returns:
    ///     True if deleted, False if not found
    fn delete_node<'a>(&self, py: Python<'a>, node_id: String) -> PyResult<Bound<'a, PyAny>> {
        let graph = self.inner.clone();

        future_into_py(py, async move {
            graph.delete_node(&node_id).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to delete node: {}", e))
            })
        })
    }
}

/// Register graph classes with Python module
pub fn register(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphNode>()?;
    m.add_class::<PyGraphRelationship>()?;
    m.add_class::<PyGraphTraversalResult>()?;
    m.add_class::<PyMemoryGraph>()?;
    Ok(())
}
