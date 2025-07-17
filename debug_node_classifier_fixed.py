#!/usr/bin/env python3
"""Debug script to test the fixed node classifier."""

import torch
import torch.nn as nn
from neurosheaf.sheaf import SheafBuilder
from neurosheaf.visualization.enhanced.node_classifier import EnhancedNodeClassifier, NodeType

# Create the same model as in test_all.py
model = nn.Sequential(                                                                                                                                         
        nn.Linear(100, 64),                                                                                                                                       
        nn.ReLU(),                                                                                                                                                 
        nn.Linear(64, 32),                                                                                                                                         
        nn.ReLU(),                                                                                                                                                
        nn.Linear(32, 10)
)

data = torch.randn(500,100)

print("=== Building Sheaf with Enhanced Metadata ===")
builder = SheafBuilder()
sheaf = builder.build_from_activations(model, data, use_gram_regularization=True, validate=False)

print(f"Sheaf metadata keys: {list(sheaf.metadata.keys())}")
print(f"Traced model available: {'traced_model' in sheaf.metadata}")
print(f"Module types available: {'module_types' in sheaf.metadata}")
print(f"Module types: {sheaf.metadata.get('module_types', {})}")

print("\n=== Testing Node Classification ===")
classifier = EnhancedNodeClassifier()

# Create model context like the visualization does
model_context = {
    'traced_model': sheaf.metadata.get('traced_model'),
    'module_types': sheaf.metadata.get('module_types', {})
}

print(f"Model context keys: {list(model_context.keys())}")
print(f"Traced model in context: {model_context['traced_model'] is not None}")
print(f"Module types in context: {len(model_context['module_types'])}")

print("\n=== Node Classification Results ===")
for node in sheaf.poset.nodes():
    node_attrs = sheaf.poset.nodes[node]
    
    print(f"\nNode: {node}")
    print(f"  Attributes: {node_attrs}")
    
    # Test classification with the fixed classifier
    node_type = classifier.classify_node(node, node_attrs, model_context)
    print(f"  Classification: {node_type}")
    
    # If it's still unknown, debug why
    if node_type.value == 'unknown':
        print(f"  DEBUG: Still unknown - checking traced model lookup...")
        if model_context['traced_model'] and node_attrs.get('op') == 'call_module':
            try:
                target = node_attrs.get('target', '')
                actual_module = model_context['traced_model'].get_submodule(target)
                print(f"  DEBUG: Target '{target}' -> {type(actual_module)}")
                module_type = type(actual_module)
                classified_type = classifier._classify_from_module_type(module_type)
                print(f"  DEBUG: Should be classified as: {classified_type}")
            except Exception as e:
                print(f"  DEBUG: Failed to get module: {e}")

print("\n=== Architecture Summary ===")
# Count node types
type_counts = {}
for node in sheaf.poset.nodes():
    node_attrs = sheaf.poset.nodes[node]
    node_type = classifier.classify_node(node, node_attrs, model_context)
    type_counts[node_type] = type_counts.get(node_type, 0) + 1

for node_type, count in type_counts.items():
    print(f"  {node_type.value.title()}: {count}")

total_unknown = type_counts.get(NodeType.UNKNOWN, 0)
total_nodes = len(sheaf.poset.nodes())
unknown_percentage = (total_unknown / total_nodes) * 100 if total_nodes > 0 else 0

print(f"\nClassification Success Rate: {100 - unknown_percentage:.1f}% ({total_nodes - total_unknown}/{total_nodes} nodes classified)")