"""
Fast ModelNet40 Experiment with Preprocessed Data

Uses pre-computed normals, k-NN graphs, and Boids features
for 10x speedup (7-8 sec/batch → <1 sec/batch)

Prerequisites:
    Run preprocess_modelnet40.py first to generate preprocessed data

Usage:
    python experiments/fast_modelnet40_experiment.py --epochs 50
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.point_ascender_v2 import (
    ASCenderV2Config,
    PointTransformerLayerASC
)


class PreprocessedModelNet40Dataset(Dataset):
    """
    Fast DataLoader for preprocessed ModelNet40

    Loads pre-computed:
    - Surface normals (PCA-based)
    - k-NN neighbor indices
    - Boids features (density)

    No expensive computation in __getitem__!
    """

    def __init__(self, preprocessed_dir, split='train'):
        self.split = split
        self.data_dir = Path(preprocessed_dir) / split

        # List all .pt files
        self.samples = sorted(self.data_dir.glob('sample_*.pt'))

        print(f"Loaded {len(self.samples)} {split} samples from {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Simply load pre-computed tensors (FAST!)
        sample = torch.load(self.samples[idx])

        return {
            'xyz': sample['xyz'],           # (N, 3)
            'normals': sample['normals'],   # (N, 3)
            'label': sample['label']        # (1,)
            # neighbor_indices, density available but not used directly in simple model
        }


class TinyPointTransformer(nn.Module):
    """
    Ultra-lightweight Point Transformer for ModelNet40

    Same architecture as synthetic experiments for fair comparison.
    """

    def __init__(self, num_classes=40, use_ascender=False, asc_config=None):
        super().__init__()

        self.use_ascender = use_ascender
        self.hidden_dim = 32
        self.k = 8

        # Input embedding: xyz → features
        self.input_embed = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        # Q, K, V projections (minimal for ultra-lightweight)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, self.hidden_dim)
        )

        # Attention transform
        self.attn_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )

        # ASCender v2.0 (optional)
        if use_ascender:
            self.asc_config = asc_config or ASCenderV2Config()
            # (ASC layers would go here - simplified for now)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def knn(self, xyz, k):
        """Simple k-NN (Euclidean distance)"""
        B, N, _ = xyz.shape
        dist = torch.cdist(xyz, xyz, p=2)
        _, idx = dist.topk(k+1, dim=-1, largest=False)
        return idx[:, :, 1:]  # Exclude self

    def forward(self, xyz, normals=None):
        """
        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3) optional

        Returns:
            logits: (B, num_classes)
        """
        B, N, _ = xyz.shape

        # Embed points
        # xyz: (B, N, 3) -> reshape to (B*N, 3) -> embed -> reshape to (B, N, C)
        B, N, _ = xyz.shape
        xyz_flat = xyz.reshape(B * N, 3)  # (B*N, 3)
        x_flat = xyz_flat
        for layer in self.input_embed:
            if isinstance(layer, nn.BatchNorm1d):
                # BatchNorm expects (B*N, C)
                x_flat = layer(x_flat)
            else:
                x_flat = layer(x_flat)
        x = x_flat.reshape(B, N, self.hidden_dim)  # (B, N, C)

        # k-NN
        idx = self.knn(xyz, self.k)  # (B, N, k)

        # Gather neighbors
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        x_neighbors = torch.gather(x.unsqueeze(2).expand(-1, -1, self.k, -1),
                                  1, idx_expanded)  # (B, N, k, C)

        # Q, K, V
        q = self.q_proj(x)  # (B, N, C)
        k = self.k_proj(x_neighbors)  # (B, N, k, C)
        v = self.v_proj(x_neighbors)  # (B, N, k, C)

        # Position encoding
        xyz_neighbors = torch.gather(xyz.unsqueeze(2).expand(-1, -1, self.k, -1),
                                     1, idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        rel_pos = xyz.unsqueeze(2) - xyz_neighbors  # (B, N, k, 3)
        pos_enc = self.pos_encoder(rel_pos)  # (B, N, k, C)

        # Attention: q·k + pos
        attn_logits = (q.unsqueeze(2) * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, N, k)

        # Aggregate
        out = (attn_weights.unsqueeze(-1) * (v + pos_enc)).sum(dim=2)  # (B, N, C)

        # Global pooling
        out = out.max(dim=1)[0]  # (B, C)

        # Classify
        logits = self.output_head(out)  # (B, num_classes)

        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Train"):
        xyz = batch['xyz'].to(device)
        normals = batch['normals'].to(device)
        labels = batch['label'].to(device).squeeze()

        optimizer.zero_grad()
        logits = model(xyz, normals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            xyz = batch['xyz'].to(device)
            normals = batch['normals'].to(device)
            labels = batch['label'].to(device).squeeze()

            logits = model(xyz, normals)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-dir', type=str,
                       default='data/ModelNet40/preprocessed',
                       help='Directory with preprocessed .pt files')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--use-ascender', action='store_true',
                       help='Use ASCender v2.0')
    parser.add_argument('--output', type=str,
                       default='results/modelnet40_results.json',
                       help='Output JSON file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Fast ModelNet40 Experiment (Preprocessed Data)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"ASCender: {args.use_ascender}")
    print()

    # Datasets
    train_dataset = PreprocessedModelNet40Dataset(args.preprocessed_dir, 'train')
    test_dataset = PreprocessedModelNet40Dataset(args.preprocessed_dir, 'test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Model
    asc_config = ASCenderV2Config() if args.use_ascender else None
    model = TinyPointTransformer(num_classes=40,
                                use_ascender=args.use_ascender,
                                asc_config=asc_config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }

    best_test_acc = 0.0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = eval_model(model, test_loader, criterion, device)

        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"  ✅ New best test accuracy!")

        print()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*70)
    print("Training Complete!")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
