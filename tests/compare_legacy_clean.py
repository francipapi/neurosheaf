#!/usr/bin/env python3
"""Direct comparison between legacy and clean poset extraction."""

import torch
import torchvision as tv
import sys
import os

# Add the project root to Python path to import legacy modules
sys.path.insert(0, '/Users/francescopapini/GitRepo/neurosheaf')

def compare_poset_implementations():
    """Compare legacy vs clean poset extraction side by side."""
    print("DIRECT COMPARISON: Legacy vs Clean Poset Extraction")
    print("=" * 70)
    
    # Load the same ResNet-18 model
    model = tv.models.resnet18(weights="IMAGENET1K_V1")
    model.eval()
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Import implementations
    try:
        from neurosheaf.sheaf.legacy.poset import FXPosetExtractor as LegacyExtractor
        print("✓ Legacy extractor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import legacy extractor: {e}")
        print("✗ Trying workaround...")
        # Try to work around import issues
        try:
            import logging
            logging.basicConfig(level=logging.WARNING)
            
            # Create a simplified logger for legacy code
            class SimpleLogger:
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
                def debug(self, msg): pass
            
            # Patch the logger
            import neurosheaf.sheaf.legacy.poset
            neurosheaf.sheaf.legacy.poset.logger = SimpleLogger()
            
            from neurosheaf.sheaf.legacy.poset import FXPosetExtractor as LegacyExtractor
            print("✓ Legacy extractor imported with workaround")
        except Exception as e2:
            print(f"✗ Complete failure to import legacy: {e2}")
            LegacyExtractor = None
    
    from neurosheaf.sheaf.extraction.fx_poset import FXPosetExtractor as CleanExtractor
    print("✓ Clean extractor imported successfully")
    
    results = {}
    
    # Test Legacy Implementation
    if LegacyExtractor:
        print(f"\n1. LEGACY IMPLEMENTATION")
        print("-" * 30)
        try:
            legacy_extractor = LegacyExtractor(handle_dynamic=True)
            legacy_poset = legacy_extractor.extract_poset(model)
            
            print(f"✓ Legacy poset: {legacy_poset.number_of_nodes()} nodes, {legacy_poset.number_of_edges()} edges")
            
            results['legacy'] = {
                'nodes': legacy_poset.number_of_nodes(),
                'edges': legacy_poset.number_of_edges(),
                'node_list': sorted(legacy_poset.nodes()),
                'poset': legacy_poset
            }
            
            print(f"Sample legacy nodes:")
            for i, node in enumerate(sorted(legacy_poset.nodes())[:10]):
                node_data = legacy_poset.nodes[node]
                print(f"  {i+1:2d}. {node:<20} | op={node_data.get('op', 'N/A')}")
            if legacy_poset.number_of_nodes() > 10:
                print(f"  ... and {legacy_poset.number_of_nodes() - 10} more")
                
        except Exception as e:
            print(f"✗ Legacy extraction failed: {e}")
            results['legacy'] = None
    else:
        print(f"\n1. LEGACY IMPLEMENTATION - SKIPPED (import failed)")
        results['legacy'] = None
    
    # Test Clean Implementation
    print(f"\n2. CLEAN IMPLEMENTATION")
    print("-" * 30)
    try:
        clean_extractor = CleanExtractor()
        clean_poset = clean_extractor.extract_poset(model)
        
        print(f"✓ Clean poset: {clean_poset.number_of_nodes()} nodes, {clean_poset.number_of_edges()} edges")
        
        results['clean'] = {
            'nodes': clean_poset.number_of_nodes(),
            'edges': clean_poset.number_of_edges(),
            'node_list': sorted(clean_poset.nodes()),
            'poset': clean_poset
        }
        
        print(f"Sample clean nodes:")
        for i, node in enumerate(sorted(clean_poset.nodes())[:10]):
            node_data = clean_poset.nodes[node]
            print(f"  {i+1:2d}. {node:<20} | op={node_data.get('op', 'N/A')}")
        if clean_poset.number_of_nodes() > 10:
            print(f"  ... and {clean_poset.number_of_nodes() - 10} more")
            
    except Exception as e:
        print(f"✗ Clean extraction failed: {e}")
        results['clean'] = None
    
    # Compare Results
    print(f"\n3. COMPARISON")
    print("-" * 30)
    
    if results['legacy'] and results['clean']:
        legacy = results['legacy']
        clean = results['clean']
        
        print(f"Node count difference: {clean['nodes'] - legacy['nodes']} (clean has {clean['nodes']}, legacy has {legacy['nodes']})")
        print(f"Edge count difference: {clean['edges'] - legacy['edges']} (clean has {clean['edges']}, legacy has {legacy['edges']})")
        
        # Find differences in node lists
        legacy_nodes = set(legacy['node_list'])
        clean_nodes = set(clean['node_list'])
        
        only_in_legacy = legacy_nodes - clean_nodes
        only_in_clean = clean_nodes - legacy_nodes
        common_nodes = legacy_nodes & clean_nodes
        
        print(f"\nNode overlap: {len(common_nodes)} common nodes")
        
        if only_in_legacy:
            print(f"\nNodes only in LEGACY ({len(only_in_legacy)}):")
            for node in sorted(only_in_legacy)[:10]:
                print(f"  - {node}")
            if len(only_in_legacy) > 10:
                print(f"  ... and {len(only_in_legacy) - 10} more")
        
        if only_in_clean:
            print(f"\nNodes only in CLEAN ({len(only_in_clean)}):")
            for node in sorted(only_in_clean)[:10]:
                print(f"  - {node}")
            if len(only_in_clean) > 10:
                print(f"  ... and {len(only_in_clean) - 10} more")
        
        # Check node naming differences
        print(f"\n4. NODE NAMING ANALYSIS")
        print("-" * 30)
        
        # Look for patterns in naming differences
        naming_patterns = {}
        for clean_node in only_in_clean:
            # Try to find corresponding legacy node
            potential_legacy = clean_node.replace('.', '_')
            if potential_legacy in only_in_legacy:
                naming_patterns[clean_node] = potential_legacy
        
        if naming_patterns:
            print(f"Found {len(naming_patterns)} potential naming pattern differences:")
            for clean_name, legacy_name in list(naming_patterns.items())[:5]:
                print(f"  Clean: {clean_name}")
                print(f"  Legacy: {legacy_name}")
                print()
        else:
            print("No obvious naming pattern differences found")
        
        return results
        
    else:
        print("Cannot compare - one or both extractions failed")
        return results

if __name__ == "__main__":
    try:
        results = compare_poset_implementations()
        print(f"\n" + "="*70)
        if results['legacy'] and results['clean']:
            print("COMPARISON COMPLETED")
            print(f"Target: 32 nodes, 38 edges")
            print(f"Legacy: {results['legacy']['nodes']} nodes, {results['legacy']['edges']} edges")
            print(f"Clean:  {results['clean']['nodes']} nodes, {results['clean']['edges']} edges")
        else:
            print("COMPARISON INCOMPLETE - Import issues with legacy code")
        print("="*70)
    except Exception as e:
        print(f"Script failed: {e}")
        import traceback
        traceback.print_exc()