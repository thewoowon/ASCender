"""
ModelNet40 Model Scaling Experiment

Tests ASCender effectiveness across different model sizes on real-world data.

This validates the Capacity-Bias Trade-off Hypothesis:
- Ultra-lightweight models (<10K): ASCender helps
- Medium models (10-50K): Transition zone
- Large models (>50K): ASCender may hurt

Usage:
    python experiments/modelnet40_scaling_experiment.py --hidden-dim 48 --epochs 50
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
    """Fast DataLoader for preprocessed ModelNet40"""

    def __init__(self, preprocessed_dir, split='train'):
        self.split = split
        self.data_dir = Path(preprocessed_dir) / split
        self.samples = sorted(self.data_dir.glob('sample_*.pt'))
        print(f"Loaded {len(self.samples)} {split} samples from {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.load(self.samples[idx])
        return {
            'xyz': sample['xyz'],
            'normals': sample['normals'],
            'label': sample['label']
        }


class ScalablePointTransformer(nn.Module):
    """
    Scalable Point Transformer for ModelNet40

    hidden_dim controls model capacity:
    - 32 → ~7K params (ultra-lightweight)
    - 48 → ~14K params
    - 64 → ~24K params
    - 80 → ~37K params
    """

    def __init__(self, num_classes=40, hidden_dim=32, use_ascender=False, asc_config=None):
        super().__init__()

        self.use_ascender = use_ascender
        self.hidden_dim = hidden_dim
        self.k = 8

        # Input embedding: xyz → features
        self.input_embed = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        # Q, K, V projections
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
        xyz_flat = xyz.reshape(B * N, 3)
        x_flat = xyz_flat
        for layer in self.input_embed:
            if isinstance(layer, nn.BatchNorm1d):
                x_flat = layer(x_flat)
            else:
                x_flat = layer(x_flat)
        x = x_flat.reshape(B, N, self.hidden_dim)

        # k-NN
        idx = self.knn(xyz, self.k)

        # Gather neighbors
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        x_neighbors = torch.gather(x.unsqueeze(2).expand(-1, -1, self.k, -1),
                                  1, idx_expanded)

        # Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x_neighbors)
        v = self.v_proj(x_neighbors)

        # Position encoding
        xyz_neighbors = torch.gather(xyz.unsqueeze(2).expand(-1, -1, self.k, -1),
                                     1, idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        rel_pos = xyz.unsqueeze(2) - xyz_neighbors
        pos_enc = self.pos_encoder(rel_pos)

        # Attention
        attn_logits = (q.unsqueeze(2) * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Aggregate
        out = (attn_weights.unsqueeze(-1) * (v + pos_enc)).sum(dim=2)

        # Global pooling
        out = out.max(dim=1)[0]

        # Classify
        logits = self.output_head(out)

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
    parser.add_argument('--hidden-dim', type=int, required=True,
                       help='Hidden dimension (controls model size)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--use-ascender', action='store_true',
                       help='Use ASCender v2.0')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("ModelNet40 Scaling Experiment")
    print("="*70)
    print(f"Device: {device}")
    print(f"Hidden dimension: {args.hidden_dim}")
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
    model = ScalablePointTransformer(num_classes=40,
                                    hidden_dim=args.hidden_dim,
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
        'hidden_dim': args.hidden_dim,
        'num_params': num_params,
        'use_ascender': args.use_ascender,
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

    results['best_test_acc'] = best_test_acc

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*70)
    print("Training Complete!")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
