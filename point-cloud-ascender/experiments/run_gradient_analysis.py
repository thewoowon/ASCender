"""
Gradient Conflict Analysis for Medium-Scale Degradation

Analyzes why ASCender hurts performance at h=48/h=64 by measuring:
1. Gradient conflict between bias path and learned path
2. Layer-wise α evolution during training
3. Attention distribution comparison

Usage:
    python experiments/run_gradient_analysis.py --hidden-dim 48 --seed 42
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.point_ascender_v2 import ASCenderV2Config


class PreprocessedModelNet40Dataset:
    """Fast DataLoader for preprocessed ModelNet40"""
    def __init__(self, preprocessed_dir, split='train'):
        from torch.utils.data import Dataset
        self.split = split
        self.data_dir = Path(preprocessed_dir) / split
        self.samples = sorted(self.data_dir.glob('sample_*.pt'))
        print(f"Loaded {len(self.samples)} {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.load(self.samples[idx])
        return {
            'xyz': sample['xyz'],
            'normals': sample['normals'],
            'label': sample['label']
        }


class GradientAnalysisModel(nn.Module):
    """Modified model for gradient analysis"""

    def __init__(self, num_classes=40, hidden_dim=48):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = 8

        # Same as baseline
        self.input_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim)
        )

        # α parameter (learnable mixing)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        # Dummy bias attention (simplified Boids)
        self.bias_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def knn(self, xyz, k):
        B, N, _ = xyz.shape
        dist = torch.cdist(xyz, xyz, p=2)
        _, idx = dist.topk(k+1, dim=-1, largest=False)
        return idx[:, :, 1:]

    def forward(self, xyz, normals=None, return_attention=False):
        B, N, _ = xyz.shape

        # Embed
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

        # Learned attention
        attn_learned_logits = (q.unsqueeze(2) * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
        attn_learned = torch.softmax(attn_learned_logits, dim=-1)

        # Bias attention (simplified distance-based)
        bias_scores = self.bias_mlp(rel_pos).squeeze(-1)
        attn_bias = torch.softmax(bias_scores, dim=-1)

        # Mix with α
        alpha = torch.sigmoid(self.alpha_logit)
        attn_final = alpha * attn_learned + (1 - alpha) * attn_bias

        # Aggregate
        out = (attn_final.unsqueeze(-1) * (v + pos_enc)).sum(dim=2)

        # Global pooling
        out = out.max(dim=1)[0]

        # Classify
        logits = self.output_head(out)

        if return_attention:
            return logits, {
                'alpha': alpha.item(),
                'attn_learned': attn_learned,
                'attn_bias': attn_bias,
                'attn_final': attn_final
            }

        return logits


def compute_gradient_conflict(model, batch, device):
    """Compute gradient conflict between learned and bias paths"""
    xyz = batch['xyz'].to(device)
    normals = batch['normals'].to(device)
    labels = batch['label'].to(device).squeeze()

    criterion = nn.CrossEntropyLoss()

    # Forward pass with α=1 (learned only)
    model.alpha_logit.data.fill_(10.0)  # sigmoid(10) ≈ 1
    logits_learned = model(xyz, normals)
    loss_learned = criterion(logits_learned, labels)

    # Gradient for learned path
    model.zero_grad()
    loss_learned.backward(retain_graph=True)
    grad_learned = {name: param.grad.clone() for name, param in model.named_parameters()
                   if param.grad is not None and 'alpha' not in name}

    # Forward pass with α=0 (bias only)
    model.alpha_logit.data.fill_(-10.0)  # sigmoid(-10) ≈ 0
    logits_bias = model(xyz, normals)
    loss_bias = criterion(logits_bias, labels)

    # Gradient for bias path
    model.zero_grad()
    loss_bias.backward()
    grad_bias = {name: param.grad.clone() for name, param in model.named_parameters()
                if param.grad is not None and 'alpha' not in name}

    # Compute cosine similarity
    cos_sim = {}
    for name in grad_learned.keys():
        if name in grad_bias:
            g_l = grad_learned[name].flatten()
            g_b = grad_bias[name].flatten()
            cos = torch.nn.functional.cosine_similarity(g_l.unsqueeze(0), g_b.unsqueeze(0)).item()
            cos_sim[name] = cos

    # Reset α to learnable state
    model.alpha_logit.data.fill_(0.0)

    return {
        'avg_cos_sim': np.mean(list(cos_sim.values())),
        'per_layer_cos_sim': cos_sim,
        'loss_learned': loss_learned.item(),
        'loss_bias': loss_bias.item()
    }


def train_with_analysis(model, train_loader, test_loader, args, device):
    """Train model while tracking gradient conflicts and α evolution"""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    analysis_results = {
        'alpha_history': [],
        'gradient_conflicts': [],
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Train", leave=False)):
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

            # Track α every 10 batches
            if batch_idx % 10 == 0:
                alpha = torch.sigmoid(model.alpha_logit).item()
                analysis_results['alpha_history'].append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'alpha': alpha
                })

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test", leave=False):
                xyz = batch['xyz'].to(device)
                normals = batch['normals'].to(device)
                labels = batch['label'].to(device).squeeze()

                logits = model(xyz, normals)
                loss = criterion(logits, labels)

                test_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        test_acc = correct / total
        test_loss = test_loss / len(test_loader)

        analysis_results['train_acc'].append(train_acc)
        analysis_results['test_acc'].append(test_acc)
        analysis_results['train_loss'].append(train_loss)
        analysis_results['test_loss'].append(test_loss)

        print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        print(f"α = {torch.sigmoid(model.alpha_logit).item():.4f}")

        # Gradient conflict analysis (every 5 epochs)
        if epoch % 5 == 0:
            print("  Analyzing gradient conflict...")
            model.train()
            sample_batch = next(iter(train_loader))
            conflict = compute_gradient_conflict(model, sample_batch, device)
            analysis_results['gradient_conflicts'].append({
                'epoch': epoch,
                **conflict
            })
            print(f"  Gradient cosine similarity: {conflict['avg_cos_sim']:.4f}")

    return analysis_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-dir', type=str,
                       default='data/ModelNet40/preprocessed')
    parser.add_argument('--hidden-dim', type=int, required=True,
                       help='Hidden dimension (48 or 64)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str,
                       default='results/gradient_analysis')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"Gradient Analysis: h{args.hidden_dim} (seed={args.seed})")
    print(f"{'='*70}\n")

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Dataset
    from torch.utils.data import Dataset
    class DS(Dataset):
        def __init__(self, preprocessed_dir, split):
            self.data_dir = Path(preprocessed_dir) / split
            self.samples = sorted(self.data_dir.glob('sample_*.pt'))
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            s = torch.load(self.samples[idx])
            return {'xyz': s['xyz'], 'normals': s['normals'], 'label': s['label']}

    train_dataset = DS(args.preprocessed_dir, 'train')
    test_dataset = DS(args.preprocessed_dir, 'test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = GradientAnalysisModel(num_classes=40, hidden_dim=args.hidden_dim).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")

    # Train with analysis
    results = train_with_analysis(model, train_loader, test_loader, args, device)

    # Add metadata
    results['hidden_dim'] = args.hidden_dim
    results['num_params'] = num_params
    results['seed'] = args.seed

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"gradient_analysis_h{args.hidden_dim}_seed{args.seed}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Analysis complete! Results saved to: {output_path}\n")


if __name__ == '__main__':
    main()
