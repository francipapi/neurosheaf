# Test Cases for FX-based Poset Extraction

## tests/test_poset_fx.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from neursheaf.sheaf.poset import NeuralNetworkPoset
import networkx as nx


class TinySkip(nn.Module):
    """Simple model with skip connection for testing"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        
    def forward(self, x):
        return x + torch.relu(self.conv(x))


class ComplexModel(nn.Module):
    """Model with multiple branches and merges"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2a = nn.Conv2d(32, 64, 3)
        self.conv2b = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2a = F.relu(self.conv2a(x1))
        x2b = F.relu(self.conv2b(x1))
        x3 = torch.cat([x2a, x2b], dim=1)
        return self.conv3(x3)


class TestPosetFX:
    """Test FX-based poset extraction"""
    
    def test_generic_fx_poset_simple(self):
        """Test FX extraction on simple skip connection model"""
        model = TinySkip()
        extractor = NeuralNetworkPoset()
        
        sample_input = torch.randn(1, 3, 32, 32)
        poset = extractor.extract_fx_poset(model, sample_input)
        
        # Should have skip edge captured
        assert poset.number_of_nodes() >= 3
        assert poset.number_of_edges() >= 2
        
        # Verify it's a DAG
        assert nx.is_directed_acyclic_graph(poset)
        
        # Check skip connection exists (non-sequential edge)
        paths = list(nx.all_simple_paths(poset, 'placeholder', 'output'))
        assert len(paths) > 1, "Skip connection not captured - should have multiple paths"
    
    def test_fx_complex_model(self):
        """Test FX extraction on model with branches"""
        model = ComplexModel()
        extractor = NeuralNetworkPoset()
        
        sample_input = torch.randn(1, 3, 64, 64)
        poset = extractor.extract_fx_poset(model, sample_input)
        
        # Should capture branching structure
        nodes_with_multiple_successors = [
            n for n in poset.nodes() 
            if poset.out_degree(n) > 1
        ]
        assert len(nodes_with_multiple_successors) >= 1, "Branching not captured"
        
        # Should capture merge point (cat operation)
        nodes_with_multiple_predecessors = [
            n for n in poset.nodes() 
            if poset.in_degree(n) > 1
        ]
        assert len(nodes_with_multiple_predecessors) >= 1, "Merge not captured"
    
    def test_fx_fallback_on_dynamic(self):
        """Test fallback when FX tracing fails"""
        
        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(10, 10) for _ in range(5)
                ])
                
            def forward(self, x):
                # Dynamic control flow that FX can't trace
                for i in range(x.size(0)):  # Loop over batch
                    x = self.layers[i % 5](x)
                return x
        
        model = DynamicModel()
        extractor = NeuralNetworkPoset()
        
        sample_input = torch.randn(3, 10)
        
        # Should fall back to simple extraction
        poset = extractor.extract_poset(model, sample_input)
        assert poset is not None
        assert nx.is_directed_acyclic_graph(poset)
    
    def test_fx_with_standard_models(self):
        """Test FX extraction works with torchvision models"""
        import torchvision.models as models
        
        # Test various architectures
        test_models = [
            models.resnet18(weights=None),
            models.mobilenet_v2(weights=None),
            models.efficientnet_b0(weights=None),
        ]
        
        extractor = NeuralNetworkPoset()
        sample_input = torch.randn(1, 3, 224, 224)
        
        for model in test_models:
            model.eval()
            poset = extractor.extract_poset(model, sample_input)
            
            assert nx.is_directed_acyclic_graph(poset)
            assert poset.number_of_nodes() > 10  # Non-trivial structure
            
            # Check topological ordering exists
            topo_order = list(nx.topological_sort(poset))
            assert len(topo_order) == poset.number_of_nodes()
    
    def test_sample_input_inference(self):
        """Test automatic sample input creation"""
        extractor = NeuralNetworkPoset()
        
        # CNN model
        cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3)
        )
        sample = extractor._create_sample_input(cnn)
        assert sample.shape == (1, 3, 224, 224)
        
        # MLP model
        mlp = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        sample = extractor._create_sample_input(mlp)
        assert sample.shape == (1, 784)


class TestPosetIntegration:
    """Test poset extraction integrates with sheaf construction"""
    
    def test_poset_sheaf_integration(self):
        """Test full pipeline with FX-based poset"""
        from neursheaf.sheaf import SheafConstructionPipeline
        
        # Create simple model
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        
        # Create dummy dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randint(0, 10, (100,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        # Run pipeline
        pipeline = SheafConstructionPipeline()
        sheaf, laplacian = pipeline.build_sheaf(model, dataloader)
        
        # Verify poset structure
        assert hasattr(sheaf, 'poset')
        assert nx.is_directed_acyclic_graph(sheaf.poset)
        
        # Verify stalks correspond to poset nodes
        poset_nodes = set(sheaf.poset.nodes())
        stalk_nodes = set(sheaf.stalks.keys())
        
        # Should have significant overlap
        overlap = poset_nodes & stalk_nodes
        assert len(overlap) >= min(len(poset_nodes), len(stalk_nodes)) * 0.5
```

## Integration with CI

Add to `.github/workflows/ci.yml`:

```yaml
- name: Test FX-based poset extraction
  run: |
    pytest tests/test_poset_fx.py -v
    # Also test with different PyTorch versions
    pip install torch==2.2.0  # Minimum for stable FX API
    pytest tests/test_poset_fx.py::TestPosetFX::test_fx_with_standard_models -v
```