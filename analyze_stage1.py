#!/usr/bin/env python3
import json
import numpy as np

# Load Stage 1 results
with open('stage1_results.json', 'r') as f:
    results = json.load(f)

# Filter valid results
valid_results = [r for r in results if 'error' not in r]
print(f'Stage 1 Analysis: {len(valid_results)} valid configurations tested')

# Sort by margin first, then separation ratio
valid_results.sort(key=lambda x: (x.get('margin', -999), x.get('separation_ratio', 0)), reverse=True)

# Summary statistics
margins = [r.get('margin', -999) for r in valid_results]
sep_ratios = [r.get('separation_ratio', 0) for r in valid_results]

print(f'\nSummary Statistics:')
print(f'  Best margin: {max(margins):+.6f}')
print(f'  Best separation: {max(sep_ratios):.4f}')
print(f'  Average margin: {np.mean(margins):+.6f}')
print(f'  Average separation: {np.mean(sep_ratios):.4f}')
print(f'  Positive margin configs: {sum(1 for m in margins if m > 0)}/{len(margins)}')

print(f'\nTop 10 Configurations:')
print(f"{'Rank':<4} {'Margin':<12} {'Sep Ratio':<10} {'λ':<6} {'w':<6} {'step':<6}")
print('-' * 50)

for i, result in enumerate(valid_results[:10], 1):
    margin = result.get('margin', -999)
    sep_ratio = result.get('separation_ratio', -999) 
    params = result['parameters']
    lambda_w = params.get('lambda_warp', 0)
    window = params.get('window_frac', 0) 
    step = params.get('step_penalty', 0)
    
    print(f'{i:<4} {margin:+10.6f} {sep_ratio:8.4f} {lambda_w:<6} {window:<6} {step:<6}')

# Parameter trends analysis
print(f'\nParameter Trends (Best 20 configs):')
top_20 = valid_results[:20]

# Lambda warp analysis
lambda_counts = {}
for r in top_20:
    lw = r['parameters']['lambda_warp']
    lambda_counts[lw] = lambda_counts.get(lw, 0) + 1

print(f'Lambda warp distribution:')
for lw in sorted(lambda_counts.keys()):
    print(f'  λ={lw}: {lambda_counts[lw]} configs')

# Window frac analysis  
window_counts = {}
for r in top_20:
    wf = r['parameters']['window_frac']
    window_counts[wf] = window_counts.get(wf, 0) + 1

print(f'Window fraction distribution:')
for wf in sorted(window_counts.keys()):
    print(f'  w={wf}: {window_counts[wf]} configs')

# Step penalty analysis
step_counts = {}
for r in top_20:
    sp = r['parameters']['step_penalty']
    step_counts[sp] = step_counts.get(sp, 0) + 1

print(f'Step penalty distribution:')
for sp in sorted(step_counts.keys()):
    print(f'  step={sp}: {step_counts[sp]} configs')

# Best vs worst comparison
best = valid_results[0]
worst = valid_results[-1]

print(f'\nBest vs Worst Configuration:')
print(f'Best:  Margin={best.get("margin", 0):+.6f}, Sep={best.get("separation_ratio", 0):.4f}')
print(f'       λ={best["parameters"]["lambda_warp"]}, w={best["parameters"]["window_frac"]}, step={best["parameters"]["step_penalty"]}')
print(f'Worst: Margin={worst.get("margin", 0):+.6f}, Sep={worst.get("separation_ratio", 0):.4f}')
print(f'       λ={worst["parameters"]["lambda_warp"]}, w={worst["parameters"]["window_frac"]}, step={worst["parameters"]["step_penalty"]}')

# Check if any positive margins achieved
positive_configs = [r for r in valid_results if r.get('margin', -999) > 0]
if positive_configs:
    print(f'\n🎉 POSITIVE MARGINS ACHIEVED: {len(positive_configs)} configurations')
    for i, config in enumerate(positive_configs[:5], 1):
        margin = config.get('margin', 0)
        sep_ratio = config.get('separation_ratio', 0)
        params = config['parameters']
        print(f'  {i}. Margin={margin:+.6f}, Sep={sep_ratio:.4f} (λ={params["lambda_warp"]}, w={params["window_frac"]}, step={params["step_penalty"]})')
else:
    print(f'\nNo positive margins yet - need enhancement features in Stage 2')
    
print(f'\nImprovement over baseline:')
baseline_margin = -0.522118
baseline_sep = 3.6386
best_margin = valid_results[0].get('margin', -999)
best_sep = valid_results[0].get('separation_ratio', 0)
print(f'  Margin: {baseline_margin:+.6f} → {best_margin:+.6f} (Δ = {best_margin - baseline_margin:+.6f})')
print(f'  Separation: {baseline_sep:.4f} → {best_sep:.4f} (Δ = {best_sep - baseline_sep:+.4f})')