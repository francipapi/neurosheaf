#!/usr/bin/env python3
"""Debug script to examine FX tracing in detail."""

import torch
import torch.nn as nn
import torch.fx as fx

# Create the same model as in test_all.py
model = nn.Sequential(                                                                                                                                         
        nn.Linear(100, 64),                                                                                                                                       
        nn.ReLU(),                                                                                                                                                 
        nn.Linear(64, 32),                                                                                                                                         
        nn.ReLU(),                                                                                                                                                
        nn.Linear(32, 10)
)

print("=== Model Structure ===")
for i, (name, module) in enumerate(model.named_modules()):
    if name:  # Skip the root module
        print(f"Module {i}: {name} -> {type(module)}")

print("\n=== FX Tracing ===")
traced = fx.symbolic_trace(model)

print("Graph code:")
print(traced.code)

print("\nNode details:")
for node in traced.graph.nodes:
    print(f"\nNode: {node.name}")
    print(f"  op: {node.op}")
    print(f"  target: {node.target}")
    print(f"  args: {node.args}")
    print(f"  kwargs: {node.kwargs}")
    
    # If it's a call_module, we can get the actual module
    if node.op == 'call_module':
        try:
            actual_module = traced.get_submodule(node.target)
            print(f"  actual_module: {type(actual_module)}")
        except:
            print(f"  actual_module: Could not retrieve")

print("\n=== Named modules access ===")
for name, module in traced.named_modules():
    if name:
        print(f"  {name}: {type(module)}")