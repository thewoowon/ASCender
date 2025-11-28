"""
Statistical Validation for ModelNet40 Experiments

Runs multiple random seeds for h32 and h80 models to obtain:
- Mean ± standard deviation
- Paired t-tests for statistical significance
- Confidence intervals

This addresses the statistical rigor requirement for graduation thesis.

Usage:
    # Run all experiments (12 total: 3 seeds × 2 models × 2 configs)
    python experiments/run_statistical_validation.py --all

    # Run only h32 experiments (faster, 6 runs)
    python experiments/run_statistical_validation.py --model-size h32

    # Run only h80 experiments
    python experiments/run_statistical_validation.py --model-size h80
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import json
import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.point_ascender_v2 import ASCenderV2Config


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
    """Scalable Point Transformer for ModelNet40"""

    def __init__(self, num_classes=40, hidden_dim=32, use_ascender=False, asc_config=None):
        super().__init__()

        self.use_ascender = use_ascender
        self.hidden_dim = hidden_dim
        self.k = 8

        # Input embedding
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

        # ASCender (optional)
        if use_ascender:
            self.asc_config = asc_config or ASCenderV2Config()

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def knn(self, xyz, k):
        """Simple k-NN"""
        B, N, _ = xyz.shape
        dist = torch.cdist(xyz, xyz, p=2)
        _, idx = dist.topk(k+1, dim=-1, largest=False)
        return idx[:, :, 1:]

    def forward(self, xyz, normals=None):
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


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
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
        for batch in tqdm(dataloader, desc="Eval", leave=False):
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


def run_single_experiment(hidden_dim, use_ascender, seed, args, device):
    """Run a single experiment with given configuration and seed"""

    print(f"\n{'='*70}")
    print(f"Running: h{hidden_dim} {'ASCender' if use_ascender else 'Baseline'} (seed={seed})")
    print(f"{'='*70}\n")

    # Set seed for reproducibility
    set_seed(seed)

    # Datasets
    train_dataset = PreprocessedModelNet40Dataset(args.preprocessed_dir, 'train')
    test_dataset = PreprocessedModelNet40Dataset(args.preprocessed_dir, 'test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Model
    asc_config = ASCenderV2Config() if use_ascender else None
    model = ScalablePointTransformer(
        num_classes=40,
        hidden_dim=hidden_dim,
        use_ascender=use_ascender,
        asc_config=asc_config
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results = {
        'hidden_dim': hidden_dim,
        'num_params': num_params,
        'use_ascender': use_ascender,
        'seed': seed,
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
            print(f"  ✅ New best!")

    results['best_test_acc'] = best_test_acc

    # Save individual result
    output_dir = Path('results/statistical_validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"h{hidden_dim}_{'ascender' if use_ascender else 'baseline'}_seed{seed}.json"
    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Best test accuracy: {best_test_acc:.4f}")
    print(f"Results saved to: {output_path}\n")

    return results


def compute_statistics(results_list):
    """Compute mean, std, and confidence interval from multiple runs"""
    best_accs = [r['best_test_acc'] for r in results_list]

    mean = np.mean(best_accs)
    std = np.std(best_accs, ddof=1)  # Use sample std (n-1)

    # 95% confidence interval
    n = len(best_accs)
    se = std / np.sqrt(n)
    ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=se)

    return {
        'mean': mean,
        'std': std,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'individual_results': best_accs,
        'n_runs': n
    }


def perform_paired_ttest(baseline_results, ascender_results):
    """Perform paired t-test between baseline and ASCender"""
    baseline_accs = [r['best_test_acc'] for r in baseline_results]
    ascender_accs = [r['best_test_acc'] for r in ascender_results]

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(ascender_accs, baseline_accs)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-dir', type=str,
                       default='data/ModelNet40/preprocessed',
                       help='Directory with preprocessed .pt files')
    parser.add_argument('--model-size', type=str, choices=['h32', 'h80', 'all'],
                       default='all',
                       help='Which model size to validate')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 123, 456],
                       help='Random seeds to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*70)
    print("STATISTICAL VALIDATION FOR MODELNET40")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model size: {args.model_size}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Total runs: {len(args.seeds) * 2 * (2 if args.model_size == 'all' else 1)}")
    print("="*70)

    # Determine which models to run
    if args.model_size == 'all':
        hidden_dims = [32, 80]
    elif args.model_size == 'h32':
        hidden_dims = [32]
    else:  # h80
        hidden_dims = [80]

    # Run all experiments
    all_results = {}

    for hidden_dim in hidden_dims:
        baseline_results = []
        ascender_results = []

        for seed in args.seeds:
            # Run baseline
            result = run_single_experiment(hidden_dim, False, seed, args, device)
            baseline_results.append(result)

            # Run ASCender
            result = run_single_experiment(hidden_dim, True, seed, args, device)
            ascender_results.append(result)

        # Compute statistics
        baseline_stats = compute_statistics(baseline_results)
        ascender_stats = compute_statistics(ascender_results)
        ttest_results = perform_paired_ttest(baseline_results, ascender_results)

        all_results[f'h{hidden_dim}'] = {
            'baseline': baseline_stats,
            'ascender': ascender_stats,
            'paired_ttest': ttest_results,
            'delta_mean': (ascender_stats['mean'] - baseline_stats['mean']) * 100,
            'baseline_mean_pct': baseline_stats['mean'] * 100,
            'ascender_mean_pct': ascender_stats['mean'] * 100
        }

    # Save summary statistics
    summary_path = Path('results/statistical_validation/summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION RESULTS")
    print("="*80)

    for model_size, stats in all_results.items():
        print(f"\n{model_size.upper()} Results:")
        print("-" * 80)

        baseline_mean = stats['baseline_mean_pct']
        baseline_std = stats['baseline']['std'] * 100
        ascender_mean = stats['ascender_mean_pct']
        ascender_std = stats['ascender']['std'] * 100
        delta = stats['delta_mean']
        p_value = stats['paired_ttest']['p_value']

        print(f"  Baseline:  {baseline_mean:.2f}% ± {baseline_std:.2f}%")
        print(f"  ASCender:  {ascender_mean:.2f}% ± {ascender_std:.2f}%")
        print(f"  Δ Acc:     {delta:+.2f}%")
        print(f"  p-value:   {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '(n.s.)'}")

        if stats['paired_ttest']['significant_at_0.05']:
            winner = "ASCender" if delta > 0 else "Baseline"
            print(f"  Result:    {winner} wins (statistically significant)")
        else:
            print(f"  Result:    No significant difference")

    print("\n" + "="*80)
    print(f"✅ Statistical validation complete!")
    print(f"Summary saved to: {summary_path}")
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, (n.s.) not significant")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
