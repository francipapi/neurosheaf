#!/usr/bin/env python3
"""
Evaluate MLP4x256 Model Accuracy on MNIST Dataset

This script evaluates the trained MLP4x256 model on the MNIST test dataset
and provides comprehensive performance metrics including accuracy, per-class
performance, confusion matrix, and inference statistics.
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import the model loader from neurosheaf
from neurosheaf.utils import load_model


class MLP4x256(nn.Module):
    """MLP with configurable layers for MNIST classification."""
    
    def __init__(self, num_layers: int = 10, hidden_dim: int = 32, num_classes: int = 10):
        super().__init__()
        # Store config so get_model_info can read it
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        dims = [28*28] + [hidden_dim] * num_layers
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.backbone(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, 
                  device: torch.device) -> Dict:
    """
    Evaluate the model on the test dataset and return comprehensive metrics.
    
    Returns:
        Dictionary containing accuracy metrics, per-class performance, and timing info
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    # Timing information
    inference_times = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Time the inference
            start_time = time.time()
            outputs = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Per-class accuracy
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
            # Store predictions and targets for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate metrics
    overall_accuracy = 100 * correct / total
    
    # Per-class accuracy
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Timing statistics
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    
    return {
        'overall_accuracy': overall_accuracy,
        'correct_predictions': correct,
        'total_samples': total,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm,
        'all_predictions': all_predictions,
        'all_targets': all_targets,
        'avg_inference_time_per_batch': avg_inference_time,
        'total_inference_time': total_inference_time,
        'samples_per_second': total / total_inference_time
    }


def compute_top_k_accuracy(model: nn.Module, test_loader: torch.utils.data.DataLoader,
                          device: torch.device, k_values: List[int] = [1, 3, 5]) -> Dict[int, float]:
    """Compute top-k accuracy for different k values."""
    model.eval()
    top_k_correct = {k: 0 for k in k_values}
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            total += targets.size(0)
            
            for k in k_values:
                _, top_k_pred = outputs.topk(k, 1, True, True)
                top_k_pred = top_k_pred.t()
                correct = top_k_pred.eq(targets.view(1, -1).expand_as(top_k_pred))
                top_k_correct[k] += correct[:k].reshape(-1).float().sum(0).item()
    
    return {k: 100 * top_k_correct[k] / total for k in k_values}


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    """Plot and optionally save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('MNIST MLP4x256 Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def save_results_to_json(results: Dict, model_path: str, output_path: str):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'model_path': model_path,
        'overall_accuracy': results['overall_accuracy'],
        'correct_predictions': results['correct_predictions'],
        'total_samples': results['total_samples'],
        'class_accuracies': results['class_accuracies'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'avg_inference_time_per_batch': results['avg_inference_time_per_batch'],
        'total_inference_time': results['total_inference_time'],
        'samples_per_second': results['samples_per_second']
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    """Main evaluation function."""
    # Configuration
    model_path = "models/mlp4layer_mnist_seed42.pth"  # Updated path based on testrun.py
    batch_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("MLP4x256 MNIST EVALUATION")
    print("="*60)
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Check if model file exists
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found at {model_path}")
        print("Available models:")
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                print(f"  - {model_file}")
        return
    
    # Load the model
    print(f"\nLoading model from {model_path}...")
    try:
        model = load_model(MLP4x256, model_path)
        model = model.to(device)
        print("✅ Model loaded successfully")
        
        # Print model info
        param_count = count_parameters(model)
        print(f"Model parameters: {param_count:,}")
        print(f"Model architecture: {model.num_layers} layers, {model.hidden_dim} hidden dim")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load MNIST test dataset
    print("\nLoading MNIST test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Number of batches: {len(test_loader)}")
    
    # Evaluate the model
    print(f"\n{'='*60}")
    print("RUNNING EVALUATION")
    print("="*60)
    
    start_time = time.time()
    results = evaluate_model(model, test_loader, device)
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall accuracy
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Accuracy: {results['overall_accuracy']:.2f}% ({results['correct_predictions']}/{results['total_samples']})")
    print(f"  Total evaluation time: {total_time:.2f} seconds")
    print(f"  Average inference time per batch: {results['avg_inference_time_per_batch']*1000:.2f} ms")
    print(f"  Samples per second: {results['samples_per_second']:.1f}")
    
    # Per-class accuracy
    print(f"\n📋 PER-CLASS ACCURACY:")
    for digit, accuracy in results['class_accuracies'].items():
        print(f"  Digit {digit}: {accuracy:.2f}%")
    
    # Top-k accuracy
    print(f"\n🎯 TOP-K ACCURACY:")
    top_k_results = compute_top_k_accuracy(model, test_loader, device, [1, 3, 5])
    for k, accuracy in top_k_results.items():
        print(f"  Top-{k}: {accuracy:.2f}%")
    
    # Confusion matrix analysis
    print(f"\n🔍 CONFUSION MATRIX ANALYSIS:")
    cm = results['confusion_matrix']
    
    # Most confused pairs
    confusion_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((i, j, cm[i, j]))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    print("  Most confused digit pairs:")
    for i, (true_digit, pred_digit, count) in enumerate(confusion_pairs[:5]):
        print(f"    {i+1}. {true_digit} → {pred_digit}: {count} times")
    
    # Classification report
    print(f"\n📈 DETAILED CLASSIFICATION REPORT:")
    report = classification_report(results['all_targets'], results['all_predictions'], 
                                 target_names=[str(i) for i in range(10)])
    print(report)
    
    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    json_path = output_dir / "mlp4x256_mnist_results.json"
    save_results_to_json(results, model_path, str(json_path))
    
    # Plot and save confusion matrix
    cm_path = output_dir / "mlp4x256_confusion_matrix.png"
    plot_confusion_matrix(results['confusion_matrix'], str(cm_path))
    
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Results saved in: {output_dir}")


if __name__ == "__main__":
    main()