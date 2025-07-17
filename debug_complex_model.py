#!/usr/bin/env python3
"""Debug script to test node classifier with a more complex model."""

import torch
import torch.nn as nn
from neurosheaf.sheaf import SheafBuilder
from neurosheaf.visualization.enhanced.node_classifier import EnhancedNodeClassifier, NodeType

# Create a more complex model with different layer types
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = ComplexModel()
data = torch.randn(32, 3, 32, 32)  # Batch of 32 RGB images

print("=== Building Sheaf with Complex Model ===")
builder = SheafBuilder()
sheaf = builder.build_from_activations(model, data, use_gram_regularization=True, validate=False)

print(f"Traced model available: {'traced_model' in sheaf.metadata}")
print(f"Module types: {len(sheaf.metadata.get('module_types', {}))}")

print("\n=== Node Classification Results ===")
classifier = EnhancedNodeClassifier()

# Create model context like the visualization does
model_context = {
    'traced_model': sheaf.metadata.get('traced_model'),
    'module_types': sheaf.metadata.get('module_types', {})
}

type_counts = {}
for node in sheaf.poset.nodes():
    node_attrs = sheaf.poset.nodes[node]
    node_type = classifier.classify_node(node, node_attrs, model_context)
    type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    print(f"Node {node}: {node_type.value} (target: {node_attrs.get('target', 'N/A')})")

print("\n=== Architecture Summary ===")
for node_type, count in type_counts.items():
    print(f"  {node_type.value.title()}: {count}")

total_unknown = type_counts.get(NodeType.UNKNOWN, 0)
total_nodes = len(sheaf.poset.nodes())
unknown_percentage = (total_unknown / total_nodes) * 100 if total_nodes > 0 else 0

print(f"\nClassification Success Rate: {100 - unknown_percentage:.1f}% ({total_nodes - total_unknown}/{total_nodes} nodes classified)")