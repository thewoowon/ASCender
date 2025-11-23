"""
Model Size Scaling Experiment - ASCender v2.0

Tests hypothesis:
- Small models (5K params): ASCender helps, α learns
- Medium models (50K params): ASCender helps less, α → higher
- Large models (200K params): ASCender ignored?, α → 1.0

Goal: Validate "inductive bias matters when capacity is limited"
"""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm

from models.point_ascender_v2 import ASCenderV2Config


# ============================================================================
# Synthetic Data (Same as before)
# ============================================================================

def generate_synthetic_pointcloud(num_points=128, num_classes=10):
    """Generate simple 3D shapes"""
    shapes = {
        0: 'sphere', 1: 'cube', 2: 'cylinder', 3: 'cone', 4: 'torus',
        5: 'pyramid', 6: 'ellipsoid', 7: 'octahedron', 8: 'prism', 9: 'random'
    }

    label = np.random.randint(0, num_classes)

    if label == 0:  # Sphere
        theta = np.random.uniform(0, 2*np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        r = 0.5
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
    elif label == 1:  # Cube
        x = np.random.uniform(-0.5, 0.5, num_points)
        y = np.random.uniform(-0.5, 0.5, num_points)
        z = np.random.uniform(-0.5, 0.5, num_points)
    elif label == 2:  # Cylinder
        theta = np.random.uniform(0, 2*np.pi, num_points)
        r = 0.3
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(-0.5, 0.5, num_points)
    elif label == 3:  # Cone
        theta = np.random.uniform(0, 2*np.pi, num_points)
        z = np.random.uniform(0, 1, num_points)
        r = 0.5 * (1 - z)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    else:  # Others (simplified)
        x = np.random.randn(num_points) * 0.3
        y = np.random.randn(num_points) * 0.3
        z = np.random.randn(num_points) * 0.3

    points = np.stack([x, y, z], axis=-1)

    # Compute normals (approximate)
    center = points.mean(axis=0)
    normals = points - center
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)

    return points.astype(np.float32), normals.astype(np.float32), label


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, num_points=128, num_classes=10):
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        points, normals, label = generate_synthetic_pointcloud(self.num_points, self.num_classes)
        return torch.from_numpy(points), torch.from_numpy(normals), label


# ============================================================================
# Scalable Point Transformer with RBP
# ============================================================================

class ScalablePointTransformerRBP(nn.Module):
    """Point Transformer with configurable size"""
    def __init__(self, input_dim=3, hidden_dim=32, num_classes=10, k=16,
                 ascender_config=None):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.ascender_config = ascender_config
        self.use_ascender = ascender_config is not None

        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # ASCender vector kernel
        if self.use_ascender:
            from models.point_ascender_v2 import VectorKernelEncoder
            self.vector_kernel = VectorKernelEncoder(
                hidden_dim=hidden_dim,
                config=ascender_config
            )
            # Learnable mixing parameter α
            self.alpha_logit = nn.Parameter(torch.zeros(1))

        # Output
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, points, normals):
        """
        points: (B, N, 3)
        normals: (B, N, 3)
        """
        B, N, _ = points.shape

        # Embed
        x = self.input_proj(points)  # (B, N, hidden_dim)

        # k-NN
        dist = torch.cdist(points, points)  # (B, N, N)
        _, indices = torch.topk(dist, self.k, largest=False, dim=-1)  # (B, N, k)

        # Gather neighbors
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        x_neigh = torch.gather(x.unsqueeze(1).expand(-1, N, -1, -1), 2, indices_exp)  # (B, N, k, C)

        p_neigh = torch.gather(points.unsqueeze(1).expand(-1, N, -1, -1), 2,
                               indices.unsqueeze(-1).expand(-1, -1, -1, 3))
        n_neigh = torch.gather(normals.unsqueeze(1).expand(-1, N, -1, -1), 2,
                               indices.unsqueeze(-1).expand(-1, -1, -1, 3))

        # Q, K, V
        q = self.q_proj(x)  # (B, N, C)
        k = self.k_proj(x_neigh)  # (B, N, k, C)
        v = self.v_proj(x_neigh)  # (B, N, k, C)

        # Position encoding
        rel_pos = points.unsqueeze(2) - p_neigh  # (B, N, k, 3)
        pos_enc = self.pos_encoder(rel_pos)  # (B, N, k, C)

        # Attention scores (LEARNED PATH)
        attn_scores_learned = ((q.unsqueeze(2) * (k + pos_enc)).sum(dim=-1) /
                              np.sqrt(self.hidden_dim))  # (B, N, k)
        attn_weights_learned = F.softmax(attn_scores_learned, dim=-1)

        # ASCender bias path
        if self.use_ascender:
            # Compute bias vector kernels
            p_i = points.unsqueeze(2).expand(-1, -1, self.k, -1)
            p_j = p_neigh
            x_i = x.unsqueeze(2).expand(-1, -1, self.k, -1)
            x_j = x_neigh
            n_i = normals.unsqueeze(2).expand(-1, -1, self.k, -1)
            n_j = n_neigh

            B_vec = self.vector_kernel(p_i, p_j, x_i, x_j, n_i, n_j)  # (B, N, k, C)

            # Bias attention scores
            attn_scores_bias = B_vec.sum(dim=-1)  # (B, N, k)
            attn_weights_bias = F.softmax(attn_scores_bias, dim=-1)

            # RBP: Mix with learnable α
            alpha = torch.sigmoid(self.alpha_logit)  # [0, 1]
            attn_weights = alpha * attn_weights_learned + (1 - alpha) * attn_weights_bias
        else:
            attn_weights = attn_weights_learned

        # Aggregate
        out = (attn_weights.unsqueeze(-1) * (v + pos_enc)).sum(dim=2)  # (B, N, C)

        # Residual + Norm
        out = self.norm(x + out)

        # Global pooling + classify
        out = out.mean(dim=1)  # (B, C)
        logits = self.classifier(out)  # (B, num_classes)

        return logits

    def get_alpha(self):
        if self.use_ascender:
            return torch.sigmoid(self.alpha_logit).item()
        return None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for points, normals, labels in loader:
        points, normals, labels = points.to(device), normals.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(points, normals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for points, normals, labels in loader:
            points, normals, labels = points.to(device), normals.to(device), labels.to(device)
            logits = model(points, normals)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


# ============================================================================
# Main Scaling Experiment
# ============================================================================

def run_scaling_experiment():
    print("="*70)
    print(" "*15 + "MODEL SCALING EXPERIMENT - ASCender v2.0")
    print("="*70)
    print("\nHypothesis: Inductive bias matters when capacity is limited")
    print("  - Small model (5K params): ASCender helps, α learns")
    print("  - Medium model (50K params): ASCender helps less, α higher")
    print("  - Large model (200K params): ASCender ignored, α → 1.0")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")

    # Dataset
    print("📊 Generating dataset...")
    train_dataset = PointCloudDataset(num_samples=2000, num_points=128, num_classes=10)
    val_dataset = PointCloudDataset(num_samples=400, num_points=128, num_classes=10)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model sizes to test
    model_configs = [
        ("Small (5K)", 32),      # hidden_dim=32 → ~5K params
        ("Medium (50K)", 128),   # hidden_dim=128 → ~50K params
        ("Large (200K)", 256),   # hidden_dim=256 → ~200K params
    ]

    # ASCender config (A+C - best from ablation)
    ascender_config = ASCenderV2Config(
        use_alignment=True,
        use_separation=False,
        use_cohesion=True,
        w_align=1.2,
        w_sep=1.2,
        w_coh=1.2,
        use_graph_reweight=False,
        use_value_modulation=False,
    )

    results = {}

    print("="*70)
    print("Starting experiments...")
    print("="*70)

    for model_name, hidden_dim in model_configs:
        print(f"\n{'='*70}")
        print(f"🚀 Testing: {model_name} (hidden_dim={hidden_dim})")
        print(f"{'='*70}")

        # Test baseline (no ASCender)
        print(f"\n📌 Baseline (No ASCender):")
        baseline_model = ScalablePointTransformerRBP(
            hidden_dim=hidden_dim,
            num_classes=10,
            ascender_config=None
        ).to(device)

        baseline_params = baseline_model.count_parameters()
        print(f"   Parameters: {baseline_params:,}")

        optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Train
        for epoch in range(50):
            train_epoch(baseline_model, train_loader, optimizer, criterion, device)

        _, baseline_acc = eval_model(baseline_model, val_loader, criterion, device)
        print(f"   Accuracy: {baseline_acc:.3f}")

        # Test with ASCender
        print(f"\n📌 With ASCender (A+C):")
        ascender_model = ScalablePointTransformerRBP(
            hidden_dim=hidden_dim,
            num_classes=10,
            ascender_config=ascender_config
        ).to(device)

        ascender_params = ascender_model.count_parameters()
        print(f"   Parameters: {ascender_params:,}")

        optimizer = torch.optim.Adam(ascender_model.parameters(), lr=1e-3)

        # Train
        for epoch in range(50):
            train_epoch(ascender_model, train_loader, optimizer, criterion, device)

        _, ascender_acc = eval_model(ascender_model, val_loader, criterion, device)
        alpha = ascender_model.get_alpha()

        print(f"   Accuracy: {ascender_acc:.3f}")
        print(f"   α: {alpha:.4f}")
        print(f"   Δ vs Baseline: {(ascender_acc - baseline_acc)*100:+.1f}%")

        # Store results
        results[model_name] = {
            'hidden_dim': hidden_dim,
            'params': int(ascender_params),
            'baseline_acc': float(baseline_acc),
            'ascender_acc': float(ascender_acc),
            'alpha': float(alpha),
            'improvement': float(ascender_acc - baseline_acc)
        }

    # Summary
    print("\n" + "="*70)
    print(" "*20 + "RESULTS SUMMARY")
    print("="*70)
    print("\n📊 MODEL SIZE vs ASCender EFFECTIVENESS:\n")
    print(f"{'Model':<20} {'Params':<12} {'Baseline':<10} {'ASCender':<10} {'Δ':<8} {'α':<8}")
    print("-"*70)

    for model_name, res in results.items():
        print(f"{model_name:<20} {res['params']:<12,} {res['baseline_acc']:<10.3f} "
              f"{res['ascender_acc']:<10.3f} {res['improvement']:+<8.3f} {res['alpha']:<8.4f}")

    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)

    # Analyze trend
    alphas = [results[k]['alpha'] for k in results.keys()]
    improvements = [results[k]['improvement'] for k in results.keys()]

    print(f"\n1️⃣  α Evolution as model grows:")
    for i, (model_name, res) in enumerate(results.items()):
        print(f"   {model_name}: α = {res['alpha']:.4f}")

    if alphas[-1] > alphas[0]:
        print(f"\n   ✅ α increases with model size ({alphas[0]:.3f} → {alphas[-1]:.3f})")
        print(f"      → Larger models rely MORE on learned path (ignore bias)")
    else:
        print(f"\n   🤔 α doesn't increase clearly")

    print(f"\n2️⃣  Improvement as model grows:")
    for i, (model_name, res) in enumerate(results.items()):
        print(f"   {model_name}: Δ = {res['improvement']:+.3f} ({res['improvement']*100:+.1f}%)")

    if improvements[-1] < improvements[0]:
        print(f"\n   ✅ Improvement decreases with model size")
        print(f"      → Small models benefit MORE from inductive bias")
    else:
        print(f"\n   🤔 Improvement doesn't decrease clearly")

    print("\n" + "="*70)
    print("💾 Results saved to results/scaling_results.json")
    print("="*70)

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_scaling_experiment()
