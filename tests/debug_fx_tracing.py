#!/usr/bin/env python3
"""Debug FX tracing to see what nodes are extracted from ResNet-18."""

import torch
import torchvision as tv
import torch.fx as fx
import operator

# Import only what we need for debugging
from neurosheaf.sheaf.extraction.fx_poset import FXPosetExtractor as CleanExtractor

def debug_clean_fx_extraction():
    """Debug clean FX extraction on ResNet-18."""
    print("Debugging Clean FX Extraction on ResNet-18")
    print("=" * 50)
    
    # Load ResNet-18
    model = tv.models.resnet18(weights="IMAGENET1K_V1")
    model.eval()
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test FX tracing
    print(f"\n1. Testing FX symbolic tracing...")
    try:
        traced = fx.symbolic_trace(model)
        print(f"   ✓ FX tracing successful")
        print(f"   ✓ Graph has {len(traced.graph.nodes)} nodes")
        
        # Show all nodes with their details
        print(f"\n   All FX nodes:")
        for i, node in enumerate(traced.graph.nodes):
            print(f"     {i+1:2d}. {node.name:<20} | op={node.op:<15} | target={str(node.target)}")
            
    except Exception as e:
        print(f"   ✗ FX tracing failed: {e}")
        return
    
    # Test clean extractor  
    print(f"\n2. Testing Clean FX Extractor...")
    try:
        clean_extractor = CleanExtractor()
        clean_poset = clean_extractor.extract_poset(model)
        
        print(f"   ✓ Clean extraction successful")
        print(f"   ✓ Poset: {clean_poset.number_of_nodes()} nodes, {clean_poset.number_of_edges()} edges")
        
        print(f"\n   Clean nodes:")
        for i, node in enumerate(clean_poset.nodes()):
            node_data = clean_poset.nodes[node]
            print(f"     {i+1:2d}. {node:<25} | op={node_data.get('op', 'N/A'):<12}")
            
    except Exception as e:
        print(f"   ✗ Clean extraction failed: {e}")
        clean_poset = None
    
    # Analyze what _is_activation_node catches
    print(f"\n3. Analyzing what _is_activation_node catches:")
    
    clean_extractor_debug = CleanExtractor()
    activation_count = 0
    non_activation_count = 0
    
    print(f"   Nodes classified as ACTIVATION nodes:")
    for node in traced.graph.nodes:
        if clean_extractor_debug._is_activation_node(node):
            activation_count += 1
            print(f"     ✓ {node.name:<20} | op={node.op:<15} | target={str(node.target)[:40]}")
    
    print(f"\n   Nodes classified as NON-ACTIVATION nodes:")
    for node in traced.graph.nodes:
        if not clean_extractor_debug._is_activation_node(node):
            non_activation_count += 1
            if non_activation_count <= 20:  # Show first 20
                print(f"     ✗ {node.name:<20} | op={node.op:<15} | target={str(node.target)[:40]}")
    
    if non_activation_count > 20:
        print(f"     ... and {non_activation_count - 20} more non-activation nodes")
    
    print(f"\n   Summary:")
    print(f"     Total FX nodes: {len(traced.graph.nodes)}")
    print(f"     Activation nodes: {activation_count}")
    print(f"     Non-activation nodes: {non_activation_count}")
    print(f"     Final poset nodes: {clean_poset.number_of_nodes() if clean_poset else 'N/A'}")
    
    # Show what we expect for ResNet-18
    print(f"\n4. Expected ResNet-18 structure:")
    print(f"   Should have ~32 nodes including:")
    print(f"     - conv1, bn1 (initial)")
    print(f"     - layer1.0.conv1, layer1.0.bn1, layer1.0.conv2, layer1.0.bn2")
    print(f"     - layer1.1.conv1, layer1.1.bn1, layer1.1.conv2, layer1.1.bn2")
    print(f"     - layer2.0.conv1, layer2.0.bn1, layer2.0.conv2, layer2.0.bn2, layer2.0.downsample")
    print(f"     - layer2.1.conv1, layer2.1.bn1, layer2.1.conv2, layer2.1.bn2")
    print(f"     - layer3.0.conv1, layer3.0.bn1, layer3.0.conv2, layer3.0.bn2, layer3.0.downsample")
    print(f"     - layer3.1.conv1, layer3.1.bn1, layer3.1.conv2, layer3.1.bn2")
    print(f"     - layer4.0.conv1, layer4.0.bn1, layer4.0.conv2, layer4.0.bn2, layer4.0.downsample")
    print(f"     - layer4.1.conv1, layer4.1.bn1, layer4.1.conv2, layer4.1.bn2")
    print(f"     - avgpool, fc (final)")
    print(f"     - Plus functional ops: relu, add operations")

if __name__ == "__main__":
    debug_clean_fx_extraction()