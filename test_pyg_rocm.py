#!/usr/bin/env python3
"""
PyTorch Geometric + ROCm Test Suite

This script tests:
1. PyTorch installation and ROCm support
2. PyTorch Geometric installation and basic functionality
3. GPU acceleration with ROCm
4. Graph neural network operations on GPU
5. Package installation locations
"""

import sys
import traceback
import os
from typing import Optional

def test_package_locations():
    """Test and display package installation locations."""
    print("=" * 60)
    print("Package Installation Locations")
    print("=" * 60)
    
    try:
        # Test PyTorch location
        import torch
        torch_location = torch.__file__
        torch_dir = os.path.dirname(torch_location)
        print(f"[OK] PyTorch location: {torch_dir}")
        print(f"     Version: {torch.__version__}")
        
        # Check if it's a conda/pip installation
        if 'conda' in torch_dir.lower():
            print("     Installation type: Conda")
        elif 'site-packages' in torch_dir.lower():
            print("     Installation type: Pip")
        else:
            print("     Installation type: Unknown/Custom")
        
        # Test PyTorch Geometric location
        try:
            import torch_geometric
            pyg_location = torch_geometric.__file__
            pyg_dir = os.path.dirname(pyg_location)
            print(f"[OK] PyTorch Geometric location: {pyg_dir}")
            print(f"     Version: {torch_geometric.__version__}")
            
            if 'conda' in pyg_dir.lower():
                print("     Installation type: Conda")
            elif 'site-packages' in pyg_dir.lower():
                print("     Installation type: Pip")
            else:
                print("     Installation type: Unknown/Custom")
                
        except ImportError:
            print("[FAIL] PyTorch Geometric not found")
            
        # Check for ROCm-specific installations
        try:
            # Check if this is a ROCm build
            if hasattr(torch.version, 'hip') and torch.version.hip:
                print(f"[OK] ROCm/HIP version: {torch.version.hip}")
            else:
                print("[WARN] No ROCm/HIP version info found")
                
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"[OK] CUDA version info: {torch.version.cuda}")
            else:
                print("[WARN] No CUDA version info found")
                
        except AttributeError:
            print("[WARN] Version information not available")
            
        # Display Python environment info
        print(f"[OK] Python executable: {sys.executable}")
        print(f"[OK] Python version: {sys.version.split()[0]}")
        
        # Check for virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print(f"[OK] Virtual environment detected: {sys.prefix}")
        else:
            print("[WARN] No virtual environment detected (using system Python)")
            
        return True
        
    except Exception as e:
        print(f"[FAIL] Package location test failed: {e}")
        traceback.print_exc()
        return False

def test_pytorch_installation():
    """Test PyTorch installation and ROCm support."""
    print("\n" + "=" * 60)
    print("Testing PyTorch Installation and ROCm Support")
    print("=" * 60)
    
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        
        # Check ROCm availability
        if torch.cuda.is_available():
            print(f"[OK] CUDA/ROCm available: {torch.cuda.is_available()}")
            print(f"[OK] Device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                print(f"[OK] Device {i}: {device_name}")
                
            # Test basic tensor operations on GPU
            device = torch.device('cuda:0')
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print(f"[OK] Basic GPU tensor operations working")
            print(f"[OK] Result tensor shape: {z.shape}, device: {z.device}")
            
        else:
            print("[WARN] CUDA/ROCm not available - will test CPU only")
            
        return torch.cuda.is_available()
        
    except Exception as e:
        print(f"[FAIL] PyTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_pytorch_geometric():
    """Test PyTorch Geometric installation and basic functionality."""
    print("\n" + "=" * 60)
    print("Testing PyTorch Geometric Installation")
    print("=" * 60)
    
    try:
        import torch_geometric
        print(f"[OK] PyTorch Geometric version: {torch_geometric.__version__}")
        
        # Test basic imports
        from torch_geometric.data import Data
        from torch_geometric.utils import to_networkx
        print("[OK] Basic imports successful")
        
        # Check for optional dependencies
        optional_deps = {
            'torch_scatter': 'Scatter operations',
            'torch_sparse': 'Sparse operations', 
            'torch_cluster': 'Clustering operations',
            'torch_spline_conv': 'Spline convolutions'
        }
        
        for dep, desc in optional_deps.items():
            try:
                __import__(dep)
                print(f"[OK] Optional dependency {dep} ({desc}): Available")
            except ImportError:
                print(f"[WARN] Optional dependency {dep} ({desc}): Not available")
        
        # Create a simple graph
        import torch
        edge_index = torch.tensor([[0, 1, 1, 2],
                                  [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        
        print(f"[OK] Created graph with {data.num_nodes} nodes and {data.num_edges} edges")
        print(f"[OK] Node features shape: {data.x.shape}")
        print(f"[OK] Edge index shape: {data.edge_index.shape}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] PyTorch Geometric test failed: {e}")
        traceback.print_exc()
        return False

def test_gnn_operations(use_gpu: bool = True):
    """Test Graph Neural Network operations."""
    print("\n" + "=" * 60)
    print("Testing GNN Operations")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, GATConv, SAGEConv
        from torch_geometric.data import Data
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
        
        # Determine device
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"[OK] Using device: {device}")
        
        # Create a simple GNN model
        class SimpleGCN(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, output_dim)
                self.dropout = torch.nn.Dropout(0.5)
                
            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.conv2(x, edge_index)
                return F.log_softmax(x, dim=1)
        
        # Test with synthetic data
        num_nodes = 1000
        num_features = 16
        num_classes = 7
        
        # Create synthetic graph data
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        x = torch.randn(num_nodes, num_features)
        y = torch.randint(0, num_classes, (num_nodes,))
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data.to(device)
        
        print(f"[OK] Created synthetic graph: {num_nodes} nodes, {data.num_edges} edges")
        
        # Initialize model
        model = SimpleGCN(num_features, 32, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        print("[OK] Model initialized and moved to device")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            out = model(data)
            print(f"[OK] Forward pass successful, output shape: {out.shape}")
            
        # Test training step
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        print(f"[OK] Training step successful, loss: {loss.item():.4f}")
        
        # Test different GNN layers
        gat_layer = GATConv(num_features, 32, heads=4, dropout=0.1).to(device)
        sage_layer = SAGEConv(num_features, 32).to(device)
        
        with torch.no_grad():
            gat_out = gat_layer(data.x, data.edge_index)
            sage_out = sage_layer(data.x, data.edge_index)
            
        print(f"[OK] GAT layer output shape: {gat_out.shape}")
        print(f"[OK] SAGE layer output shape: {sage_out.shape}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] GNN operations test failed: {e}")
        traceback.print_exc()
        return False

def test_real_dataset():
    """Test with a real dataset from PyTorch Geometric."""
    print("\n" + "=" * 60)
    print("Testing Real Dataset (Cora)")
    print("=" * 60)
    
    try:
        import torch
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
        
        # Load Cora dataset
        dataset = Planetoid(root='/tmp/Cora', name='Cora', 
                           transform=NormalizeFeatures())
        data = dataset[0]
        
        print(f"[OK] Loaded Cora dataset")
        print(f"[OK] Number of graphs: {len(dataset)}")
        print(f"[OK] Number of features: {dataset.num_features}")
        print(f"[OK] Number of classes: {dataset.num_classes}")
        print(f"[OK] Number of nodes: {data.num_nodes}")
        print(f"[OK] Number of edges: {data.num_edges}")
        print(f"[OK] Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"[OK] Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"[OK] Has self loops: {data.has_self_loops()}")
        print(f"[OK] Is undirected: {data.is_undirected()}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Real dataset test failed: {e}")
        traceback.print_exc()
        return False

def test_gpu_memory_usage():
    """Test GPU memory usage with PyTorch Geometric."""
    print("\n" + "=" * 60)
    print("Testing GPU Memory Usage")
    print("=" * 60)
    
    try:
        import torch
        from torch_geometric.data import Data
        
        if not torch.cuda.is_available():
            print("[WARN] GPU not available, skipping memory test")
            return True
            
        device = torch.device('cuda:0')
        
        # Check initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        print(f"[OK] Initial GPU memory: {initial_memory / 1024**2:.2f} MB")
        
        # Create large graph
        num_nodes = 10000
        num_edges = 50000
        num_features = 128
        
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        x = torch.randn(num_nodes, num_features, device=device)
        
        data = Data(x=x, edge_index=edge_index)
        
        current_memory = torch.cuda.memory_allocated(device)
        print(f"[OK] Memory after creating large graph: {current_memory / 1024**2:.2f} MB")
        print(f"[OK] Memory used by graph: {(current_memory - initial_memory) / 1024**2:.2f} MB")
        
        # Clean up
        del data, x, edge_index
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(device)
        print(f"[OK] Memory after cleanup: {final_memory / 1024**2:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] GPU memory test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("PyTorch Geometric + ROCm Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test package locations first
    results['package_locations'] = test_package_locations()
    
    # Test PyTorch and ROCm
    results['pytorch'] = test_pytorch_installation()
    gpu_available = results['pytorch']
    
    # Test PyTorch Geometric
    results['pyg'] = test_pytorch_geometric()
    
    # Test GNN operations
    results['gnn'] = test_gnn_operations(use_gpu=gpu_available)
    
    # Test real dataset
    results['dataset'] = test_real_dataset()
    
    # Test GPU memory (only if GPU available)
    if gpu_available:
        results['gpu_memory'] = test_gpu_memory_usage()
    else:
        results['gpu_memory'] = True  # Skip this test
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name.upper().replace('_', ' '):<20}: [{status}]")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: ALL TESTS PASSED! PyTorch Geometric with ROCm is working correctly.")
    else:
        print("WARNING: SOME TESTS FAILED. Check the output above for details.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)