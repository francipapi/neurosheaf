"""Deterministic indexing utilities for consistent matrix construction.

This module provides canonical ordering functions to ensure reproducible
matrix layouts across different runs and architectures. The key insight is
that NetworkX node/edge iteration order is non-deterministic, so we must
always use sorted canonical orderings for matrix construction.

Mathematical Importance:
- Different node ordering → different block layout → numerically different matrices
- This breaks cross-architecture comparability even for identical inputs  
- Canonical ordering ensures bitwise-identical results for reproducible research
"""

from typing import List, Dict, Tuple, Any, Callable, Union
import logging

logger = logging.getLogger(__name__)


def canonical_node_order(nodes, key: Callable = str) -> List:
    """Return nodes in canonical sorted order.
    
    Uses string representation for sorting to ensure deterministic ordering
    across different Python versions and architectures.
    
    Args:
        nodes: Iterable of node identifiers
        key: Function to convert nodes to sortable form (default: str)
        
    Returns:
        List of nodes in canonical sorted order
    """
    return sorted(nodes, key=key)


def canonical_edge_order(edges, key: Callable = None) -> List[Tuple]:
    """Return edges in canonical sorted order.
    
    Default ordering sorts by (str(source), str(target)) to ensure
    deterministic edge ordering for matrix construction.
    
    Args:
        edges: Iterable of (source, target) edge tuples
        key: Function to convert edges to sortable form
        
    Returns:
        List of edges in canonical sorted order
    """
    if key is None:
        key = lambda e: (str(e[0]), str(e[1]))
    
    return sorted(edges, key=key)


def create_node_mapping(nodes) -> Tuple[Dict, Dict, List]:
    """Create bidirectional node mappings with canonical ordering.
    
    Args:
        nodes: Iterable of node identifiers
        
    Returns:
        node2idx: Dictionary mapping node -> index
        idx2node: Dictionary mapping index -> node  
        canonical_nodes: List of nodes in canonical order
    """
    canonical_nodes = canonical_node_order(nodes)
    node2idx = {node: i for i, node in enumerate(canonical_nodes)}
    idx2node = {i: node for i, node in enumerate(canonical_nodes)}
    
    return node2idx, idx2node, canonical_nodes


def create_edge_mapping(edges) -> Tuple[Dict, Dict, List]:
    """Create bidirectional edge mappings with canonical ordering.
    
    Args:
        edges: Iterable of (source, target) edge tuples
        
    Returns:
        edge2idx: Dictionary mapping edge -> index
        idx2edge: Dictionary mapping index -> edge
        canonical_edges: List of edges in canonical order
    """
    canonical_edges = canonical_edge_order(edges)
    edge2idx = {edge: i for i, edge in enumerate(canonical_edges)}
    idx2edge = {i: edge for i, edge in enumerate(canonical_edges)}
    
    return edge2idx, idx2edge, canonical_edges


def validate_round_trip_mapping(mapping: Dict, items: List, mapping_name: str = "mapping") -> None:
    """Validate that mapping is bijective (round-trip identity).
    
    Ensures that item -> index -> item produces the original item for all items.
    This catches ordering inconsistencies that would break matrix construction.
    
    Args:
        mapping: Dictionary mapping items to indices
        items: List of items in canonical order
        mapping_name: Name for error messages
        
    Raises:
        AssertionError: If round-trip fails for any item
    """
    # Check size first for clearer error messages
    assert len(mapping) == len(items), \
        f"{mapping_name} size mismatch: {len(mapping)} != {len(items)}"
    
    # Then check round-trip property
    for i, item in enumerate(items):
        mapped_idx = mapping.get(item)
        assert mapped_idx == i, \
            f"Round-trip failed for {mapping_name}: {item} -> {mapped_idx} != {i}"
    
    logger.debug(f"✅ Validated round-trip mapping for {len(items)} {mapping_name} items")


def create_sheaf_indexing_metadata(poset) -> Dict[str, Any]:
    """Create complete indexing metadata for a sheaf.
    
    This is the primary function that should be called by builders to
    create consistent indexing metadata for sheaf objects.
    
    Args:
        poset: NetworkX graph (DiGraph or Graph)
        
    Returns:
        Dictionary with canonical ordering and mapping metadata
    """
    # Create canonical node ordering and mapping
    node2idx, idx2node, canonical_nodes = create_node_mapping(poset.nodes())
    
    # Create canonical edge ordering and mapping  
    edge2idx, idx2edge, canonical_edges = create_edge_mapping(poset.edges())
    
    # Validate mappings
    validate_round_trip_mapping(node2idx, canonical_nodes, "node2idx")
    validate_round_trip_mapping(edge2idx, canonical_edges, "edge2idx")
    
    logger.info(f"Created canonical indexing: {len(canonical_nodes)} nodes, {len(canonical_edges)} edges")
    
    return {
        # Canonical orderings
        'node_order': canonical_nodes,
        'edge_order': canonical_edges,
        
        # Forward mappings (item -> index)
        'node2idx': node2idx,
        'edge2idx': edge2idx,
        
        # Reverse mappings (index -> item)  
        'idx2node': idx2node,
        'idx2edge': idx2edge,
        
        # Versioning for future compatibility
        'indexing_version': '1.0',
        'ordering_method': 'canonical_string_sort'
    }


def validate_sheaf_indexing(sheaf) -> bool:
    """Validate that sheaf metadata contains valid deterministic indexing.
    
    This should be called after sheaf construction to ensure indexing
    metadata is present and consistent.
    
    Args:
        sheaf: Sheaf object with metadata
        
    Returns:
        True if indexing is valid
        
    Raises:
        AssertionError: If indexing is invalid or missing
    """
    required_keys = ['node_order', 'edge_order', 'node2idx', 'edge2idx']
    
    for key in required_keys:
        assert key in sheaf.metadata, f"Missing indexing metadata: {key}"
    
    # Extract metadata
    nodes = sheaf.metadata['node_order']
    edges = sheaf.metadata['edge_order']
    node2idx = sheaf.metadata['node2idx']
    edge2idx = sheaf.metadata['edge2idx']
    
    # Validate round-trip mappings
    validate_round_trip_mapping(node2idx, nodes, "node2idx")
    validate_round_trip_mapping(edge2idx, edges, "edge2idx")
    
    # Validate completeness against poset
    assert set(nodes) == set(sheaf.poset.nodes()), \
        "Node ordering doesn't match poset nodes"
    assert set(edges) == set(sheaf.poset.edges()), \
        "Edge ordering doesn't match poset edges"
    
    # Validate canonical ordering (should be sorted)
    assert nodes == canonical_node_order(nodes), \
        "Node order is not canonical"
    assert edges == canonical_edge_order(edges), \
        "Edge order is not canonical"
    
    logger.debug("✅ Validated sheaf indexing metadata")
    return True


def get_legacy_ordering_from_metadata(sheaf) -> Tuple[List, List]:
    """Extract legacy (potentially non-canonical) ordering from metadata.
    
    For debugging and migration purposes. Allows comparison between
    old non-deterministic and new canonical orderings.
    
    Args:
        sheaf: Sheaf object with metadata
        
    Returns:
        legacy_nodes: Original node ordering (if available)
        legacy_edges: Original edge ordering (if available)
    """
    legacy_nodes = sheaf.metadata.get('old_node_order', list(sheaf.poset.nodes()))
    legacy_edges = sheaf.metadata.get('old_edge_order', list(sheaf.poset.edges()))
    
    return legacy_nodes, legacy_edges


def compare_orderings(old_nodes: List, old_edges: List, 
                     new_nodes: List, new_edges: List) -> Dict[str, Any]:
    """Compare old vs new orderings to understand changes.
    
    Useful for debugging when transitioning to canonical ordering.
    
    Args:
        old_nodes: Previous node ordering
        old_edges: Previous edge ordering  
        new_nodes: Canonical node ordering
        new_edges: Canonical edge ordering
        
    Returns:
        Dictionary with comparison statistics
    """
    return {
        'nodes_changed': old_nodes != new_nodes,
        'edges_changed': old_edges != new_edges,
        'node_permutation': [old_nodes.index(node) for node in new_nodes] if old_nodes != new_nodes else None,
        'edge_permutation': [old_edges.index(edge) for edge in new_edges] if old_edges != new_edges else None,
        'same_node_set': set(old_nodes) == set(new_nodes),
        'same_edge_set': set(old_edges) == set(new_edges)
    }