#!/usr/bin/env python3
"""
Script to rename models in a consistent way based on architecture and training type.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_models(models_dir: Path) -> Dict[str, List[str]]:
    """Analyze existing models and categorize them."""
    categories = {
        'mlp_random': [],
        'custom_random': [],
        'torch_parallel': [],
        'torch_custom': []
    }
    
    for file_path in models_dir.glob('*.pth'):
        filename = file_path.name
        
        if filename.startswith('mlp_random'):
            categories['mlp_random'].append(filename)
        elif filename.startswith('custom_random'):
            categories['custom_random'].append(filename)
        elif filename.startswith('torch_parallel'):
            categories['torch_parallel'].append(filename)
        elif filename.startswith('torch_custom'):
            categories['torch_custom'].append(filename)
    
    return categories

def create_rename_mappings(categories: Dict[str, List[str]]) -> Dict[str, str]:
    """Create mapping from old names to new names."""
    mappings = {}
    
    # Handle MLP random models
    mlp_random = sorted(categories['mlp_random'])
    for i, old_name in enumerate(mlp_random):
        if 'seed42' in old_name:
            new_name = 'mlp_random_v00.pth'
        else:
            # Extract version number
            version_match = re.search(r'_v(\d+)\.pth', old_name)
            if version_match:
                version = version_match.group(1)
                new_name = f'mlp_random_v{version}.pth'
            else:
                new_name = f'mlp_random_v{i:02d}.pth'
        mappings[old_name] = new_name
    
    # Handle Custom random models
    custom_random = sorted(categories['custom_random'])
    for i, old_name in enumerate(custom_random):
        if 'seed42' in old_name:
            new_name = 'custom_random_v00.pth'
        else:
            # Extract version number
            version_match = re.search(r'_v(\d+)\.pth', old_name)
            if version_match:
                version = version_match.group(1)
                new_name = f'custom_random_v{version}.pth'
            else:
                new_name = f'custom_random_v{i:02d}.pth'
        mappings[old_name] = new_name
    
    # Handle torch_parallel (MLP trained) models - group by base name, then handle copies
    torch_parallel = sorted(categories['torch_parallel'])
    mlp_version = 1
    
    # Group by base filename (without " copy" suffix)
    parallel_groups = {}
    for filename in torch_parallel:
        base_name = re.sub(r'( copy \d+| copy)', '', filename)
        if base_name not in parallel_groups:
            parallel_groups[base_name] = []
        parallel_groups[base_name].append(filename)
    
    for base_name, variants in parallel_groups.items():
        # Sort variants: original first, then copies
        variants_sorted = sorted(variants, key=lambda x: (
            ' copy' in x,  # Original files first
            x.count('copy'),  # Then by copy number
            x
        ))
        
        for variant in variants_sorted:
            # Extract accuracy from base filename
            acc_match = re.search(r'acc_([0-9.]+)', base_name)
            accuracy = acc_match.group(1) if acc_match else '1.0000'
            # Remove any trailing period
            accuracy = accuracy.rstrip('.')
            
            new_name = f'mlp_trained_v{mlp_version:02d}_acc{accuracy}_ep20.pth'
            mappings[variant] = new_name
            mlp_version += 1
    
    # Handle torch_custom models - sort by the trailing float value
    torch_custom = categories['torch_custom']
    
    # Extract trailing float and sort by it
    custom_with_floats = []
    for filename in torch_custom:
        float_match = re.search(r'_([0-9.]+)\.pth$', filename)
        if float_match:
            float_val = float(float_match.group(1))
            custom_with_floats.append((float_val, filename))
    
    custom_with_floats.sort()  # Sort by float value
    
    for i, (float_val, old_name) in enumerate(custom_with_floats, 1):
        # Extract accuracy
        acc_match = re.search(r'acc_([0-9.]+)', old_name)
        accuracy = acc_match.group(1) if acc_match else '1.0000'
        
        new_name = f'custom_trained_v{i:02d}_acc{accuracy}_ep20.pth'
        mappings[old_name] = new_name
    
    return mappings

def print_rename_plan(mappings: Dict[str, str]):
    """Print the renaming plan for review."""
    print("RENAMING PLAN:")
    print("=" * 80)
    
    categories = {
        'MLP Random': [],
        'Custom Random': [],
        'MLP Trained': [],
        'Custom Trained': []
    }
    
    for old_name, new_name in mappings.items():
        if new_name.startswith('mlp_random'):
            categories['MLP Random'].append((old_name, new_name))
        elif new_name.startswith('custom_random'):
            categories['Custom Random'].append((old_name, new_name))
        elif new_name.startswith('mlp_trained'):
            categories['MLP Trained'].append((old_name, new_name))
        elif new_name.startswith('custom_trained'):
            categories['Custom Trained'].append((old_name, new_name))
    
    for category, items in categories.items():
        if items:
            print(f"\n{category} ({len(items)} files):")
            print("-" * 40)
            for old_name, new_name in sorted(items):
                print(f"  {old_name}")
                print(f"    -> {new_name}")

def execute_renames(models_dir: Path, mappings: Dict[str, str], dry_run: bool = True):
    """Execute the file renames."""
    if dry_run:
        print(f"\nDRY RUN - Would rename {len(mappings)} files")
        return
    
    print(f"\nExecuting {len(mappings)} renames...")
    
    for old_name, new_name in mappings.items():
        old_path = models_dir / old_name
        new_path = models_dir / new_name
        
        if old_path.exists():
            if new_path.exists():
                print(f"WARNING: Target exists, skipping: {new_name}")
                continue
            
            old_path.rename(new_path)
            print(f"Renamed: {old_name} -> {new_name}")
        else:
            print(f"WARNING: Source not found: {old_name}")

def main():
    models_dir = Path('models')
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    print(f"Analyzing models in: {models_dir}")
    categories = analyze_models(models_dir)
    
    total_files = sum(len(files) for files in categories.values())
    print(f"Found {total_files} total model files:")
    for category, files in categories.items():
        print(f"  {category}: {len(files)} files")
    
    # Create rename mappings
    mappings = create_rename_mappings(categories)
    
    # Print plan
    print_rename_plan(mappings)
    
    # Show summary
    print(f"\nSUMMARY:")
    print(f"Total files to rename: {len(mappings)}")
    print(f"Total files found: {total_files}")
    
    if len(mappings) != total_files:
        print("WARNING: Mismatch between files found and rename mappings!")
        return
    
    # Execute the renaming
    execute_renames(models_dir, mappings, dry_run=False)
    print("Renaming complete!")

if __name__ == "__main__":
    main()