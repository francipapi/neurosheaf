#!/usr/bin/env python3
"""Debug script to examine node classifier data."""

import torch
import torch.nn as nn
from neurosheaf.sheaf import SheafBuilder
from neurosheaf.visualization.enhanced.node_classifier import EnhancedNodeClassifier

# Create the same model as in test_all.py
model = nn.Sequential(                                                                                                                                         
        nn.Linear(100, 64),                                                                                                                                       
        nn.ReLU(),                                                                                                                                                 
        nn.Linear(64, 32),                                                                                                                                         
        nn.ReLU(),                                                                                                                                                
        nn.Linear(32, 10)
)

data = torch.randn(500,100)

builder = SheafBuilder()
sheaf = builder.build_from_activations(model, data, use_gram_regularization=True, validate=False)

print("=== Node Classifier Debug ===")
classifier = EnhancedNodeClassifier()

for node in sheaf.poset.nodes():
    node_attrs = sheaf.poset.nodes[node]
    
    print(f"\nNode: {node}")
    print(f"  Attributes: {node_attrs}")
    
    # Check what gets passed to the classifier
    op_type = node_attrs.get('op', '').lower()
    target = str(node_attrs.get('target', '')).lower()
    combined_text = f"{node} {target} {op_type}".lower()
    
    print(f"  op: '{op_type}'")
    print(f"  target: '{target}'")
    print(f"  combined_text: '{combined_text}'")
    
    # Try to classify
    node_type = classifier.classify_node(node, node_attrs)
    print(f"  Classification: {node_type}")
    
    # Check if any patterns would match
    for pattern_type, patterns in classifier.patterns.items():
        for pattern in patterns:
            import re
            if re.search(pattern, combined_text):
                print(f"  Would match {pattern_type} with pattern: {pattern}")