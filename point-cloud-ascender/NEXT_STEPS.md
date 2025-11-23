# 🎯 다음 단계 실행 계획

**작성일**: 2025-11-11
**현재 상태**: Phase 1 완료 ✅

---

## ✅ 지금까지 완료한 것

### 1. 연구 조사 & 차별점 증명
- ✅ Point Transformer 논문 분석
- ✅ 6가지 핵심 차별점 문서화
- ✅ "단순하지 않다" 증명 완료

### 2. 핵심 코드 구현
- ✅ `PointAscenderBias` 모듈 (440+ lines)
- ✅ 3D Euclidean distance, Normal alignment
- ✅ Per-head adaptive σ, γ

### 3. Baseline 준비
- ✅ Point Transformer official code 다운로드
- ✅ 구조 파악 (`PointTransformerLayer`)
- ✅ MSRAction3D 데이터셋 정보 확인

---

## 🚀 Phase 2: Quick PoC 실행 계획

### Option A: 빠른 검증 (추천) ⭐
**목표**: 2-3일 안에 "작동 가능성" 확인

**단계:**
1. **Minimal Integration** (오늘)
   - Point Transformer의 `PointTransformerLayer`에 ASCender bias 추가
   - Residual Bias Path (RBP) 구현
   - 테스트 스크립트 작성

2. **Synthetic 데이터로 빠른 테스트** (내일)
   ```python
   # 가짜 point cloud 생성
   xyz = torch.rand(B, N, 3) * 2.0  # 0~2 meters
   normals = F.normalize(torch.randn(B, N, 3))
   labels = torch.randint(0, 20, (B,))  # 20 actions

   # ASCender bias 작동 확인
   # α 값이 학습되는지 확인
   # 컴포넌트 ablation (A, S, C)
   ```

3. **결과 분석** (모레)
   - α 값이 0.5에서 벗어나는가?
   - 컴포넌트별 차이가 있는가?
   - → **Yes**: MSRAction3D로 full 실험
   - → **No**: 바이어스 강도 증가 후 재시도

**장점:**
- ⚡ 빠름 (데이터 다운로드 불필요)
- 🔧 디버깅 쉬움
- 📊 메커니즘 검증에 집중

**단점:**
- 실제 데이터 아님
- 논문 결과로는 약함

---

### Option B: 실제 데이터 (시간 소요)
**목표**: MSRAction3D로 완전한 실험

**단계:**
1. **데이터 준비** (2-3일)
   - MSRAction3D 다운로드
   - Depth → Point Cloud 변환
   - Normal estimation
   - Train/Test split

2. **모델 통합** (1-2일)
   - Point Transformer + ASCender
   - Training script 작성
   - Evaluation script

3. **실험 실행** (3-4일)
   - Baseline training
   - ASCender training
   - RBP training
   - Ablation studies

**장점:**
- 📈 논문 결과로 사용 가능
- 🎓 완전한 검증

**단점:**
- ⏰ 시간 소요 (총 7-10일)
- 💾 데이터 준비 복잡
- 🐛 디버깅 어려움

---

## 💡 **추천: Hybrid 접근**

### Week 1: Option A (Synthetic)
```bash
# Day 1 (오늘)
cd point-cloud-ascender
python experiments/test_ascender_synthetic.py
# α 학습되는지 확인

# Day 2
python experiments/ablation_synthetic.py
# A, S, C 컴포넌트 분해

# Day 3
# 결과 분석 → 방향 결정
```

**판단 기준:**
- ✅ α가 0.4~0.6 범위 벗어남 → **유망**
- ✅ A/S/C 컴포넌트별 차이 보임 → **유망**
- ❌ α ≈ 0.5 고정 → **바이어스 강도 증가**

### Week 2: Option B (Real Data) - 유망할 경우만
```bash
# MSRAction3D 다운로드
# Full 실험 진행
```

---

## 📝 지금 바로 실행 가능한 코드

### 1. Minimal Integration Script

**파일**: `experiments/test_ascender_minimal.py`

```python
"""
Minimal test: ASCender bias on synthetic point clouds
"""
import torch
import torch.nn.functional as F
import sys
sys.path.append('../src')

from models.point_ascender_bias import PointAscenderBias, PointAscenderConfig

def test_minimal():
    # Setup
    B, N, d_model = 4, 256, 64
    n_heads, d_head = 8, d_model // n_heads
    device = torch.device('cpu')  # or 'cuda'

    # Config
    cfg = PointAscenderConfig(
        use_alignment=True,
        use_separation=True,
        use_cohesion=True,
        per_head_scale=True,
        w_align=0.3,
        w_sep=0.15,
        w_coh=0.25,
    )

    # Model
    biaser = PointAscenderBias(cfg, n_heads=n_heads).to(device)

    # Synthetic data
    xyz = torch.rand(B, N, 3, device=device) * 2.0  # 0~2 meters
    normals = F.normalize(torch.randn(B, N, 3, device=device), dim=-1)
    qh = torch.randn(B, n_heads, N, d_head, device=device)
    kh = torch.randn(B, n_heads, N, d_head, device=device)

    # Forward
    bias = biaser(qh, kh, xyz, xyz, normals, normals)

    # Results
    print(f"✅ Bias shape: {bias.shape}")
    print(f"📊 Bias stats:")
    print(f"   Mean: {bias.mean().item():.4f}")
    print(f"   Std:  {bias.std().item():.4f}")
    print(f"   Min:  {bias.min().item():.4f}")
    print(f"   Max:  {bias.max().item():.4f}")

    # Per-head analysis
    gamma = biaser._get_gamma()
    sigma_sep, sigma_coh = biaser._get_sigmas()
    print(f"\n🔧 Per-head parameters:")
    print(f"   γ (scale):  {gamma.cpu().numpy()}")
    print(f"   σ_sep:      {sigma_sep.cpu().numpy()}")
    print(f"   σ_coh:      {sigma_coh.cpu().numpy()}")

if __name__ == "__main__":
    test_minimal()
```

### 2. Residual Bias Path Test

**파일**: `experiments/test_rbp_learning.py`

```python
"""
Test: Does α learn away from 0.5?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SimplePointClassifier(nn.Module):
    def __init__(self, n_classes=20, use_ascender=False):
        super().__init__()
        self.use_ascender = use_ascender

        # Simple embedding
        self.embedding = nn.Linear(3, 64)

        # ASCender bias (if enabled)
        if use_ascender:
            from models.point_ascender_bias import PointAscenderBias, PointAscenderConfig
            cfg = PointAscenderConfig(
                use_alignment=True,
                use_separation=True,
                use_cohesion=True,
                per_head_scale=True,
            )
            self.biaser = PointAscenderBias(cfg, n_heads=8)
            # Learnable α for RBP
            self.alpha_logit = nn.Parameter(torch.zeros(8))  # per head

        # Simple self-attention
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(64, 64)
        self.v = nn.Linear(64, 64)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, xyz, normals):
        B, N, _ = xyz.shape

        # Embed
        x = self.embedding(xyz)  # (B, N, 64)

        # Self-attention with ASCender
        qh = self.q(x).view(B, N, 8, 8).transpose(1, 2)  # (B, 8, N, 8)
        kh = self.k(x).view(B, N, 8, 8).transpose(1, 2)
        vh = self.v(x).view(B, N, 8, 8).transpose(1, 2)

        # Learned attention
        attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) / (8 ** 0.5)

        # ASCender bias
        if self.use_ascender:
            bias = self.biaser(qh, kh, xyz, xyz, normals, normals)

            # Residual Bias Path: α * learned + (1-α) * bias
            alpha = torch.sigmoid(self.alpha_logit).view(1, -1, 1, 1)
            attn_scores = alpha * attn_scores + (1 - alpha) * bias

        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, vh)  # (B, 8, N, 8)

        # Pool and classify
        out = out.transpose(1, 2).contiguous().view(B, N, 64)
        out = out.mean(dim=1)  # Global average pooling
        logits = self.classifier(out)

        return logits

    def get_alpha(self):
        if self.use_ascender:
            return torch.sigmoid(self.alpha_logit).detach().cpu()
        return None

def train_and_analyze():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Models
    baseline = SimplePointClassifier(n_classes=20, use_ascender=False).to(device)
    ascender = SimplePointClassifier(n_classes=20, use_ascender=True).to(device)

    # Synthetic data
    def generate_batch(batch_size=16, n_points=256):
        xyz = torch.rand(batch_size, n_points, 3, device=device) * 2.0
        normals = F.normalize(torch.randn(batch_size, n_points, 3, device=device), dim=-1)
        labels = torch.randint(0, 20, (batch_size,), device=device)
        return xyz, normals, labels

    # Training loop
    criterion = nn.CrossEntropyLoss()
    opt_baseline = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    opt_ascender = torch.optim.Adam(ascender.parameters(), lr=1e-3)

    print("🚀 Training on synthetic data...\n")

    for epoch in range(20):
        # Baseline
        opt_baseline.zero_grad()
        xyz, normals, labels = generate_batch()
        logits_baseline = baseline(xyz, normals)
        loss_baseline = criterion(logits_baseline, labels)
        loss_baseline.backward()
        opt_baseline.step()

        # ASCender
        opt_ascender.zero_grad()
        xyz, normals, labels = generate_batch()
        logits_ascender = ascender(xyz, normals)
        loss_ascender = criterion(logits_ascender, labels)
        loss_ascender.backward()
        opt_ascender.step()

        # Log
        if epoch % 5 == 0:
            alpha = ascender.get_alpha()
            print(f"Epoch {epoch:02d} | Loss: Baseline={loss_baseline.item():.3f}, ASCender={loss_ascender.item():.3f}")
            print(f"           | α (mean): {alpha.mean().item():.4f}, α (std): {alpha.std().item():.4f}")
            print(f"           | α (min):  {alpha.min().item():.4f}, α (max): {alpha.max().item():.4f}\n")

    # Final analysis
    print("="*60)
    print("📊 FINAL ANALYSIS")
    print("="*60)
    alpha_final = ascender.get_alpha()
    print(f"α values: {alpha_final.numpy()}")
    print(f"\n🎯 Interpretation:")
    mean_alpha = alpha_final.mean().item()
    if mean_alpha > 0.6:
        print("  ✅ Learned attention dominates (α > 0.6)")
        print("  → Model prefers learned patterns over spatial structure")
    elif mean_alpha < 0.4:
        print("  ✅ Spatial bias dominates (α < 0.4)")
        print("  → Boids-inspired bias provides strong value!")
    else:
        print("  ⚠️  Balanced mixing (0.4 < α < 0.6)")
        print("  → Need stronger bias OR more training")

if __name__ == "__main__":
    train_and_analyze()
```

---

## 🎯 **추천 실행 순서**

### 오늘 (30분):
```bash
cd /Users/aepeul/ASCender/point-cloud-ascender
mkdir -p experiments

# Test 1: Minimal (바이어스 작동 확인)
python experiments/test_ascender_minimal.py

# Test 2: RBP Learning (α 학습 확인)
python experiments/test_rbp_learning.py
```

**예상 결과:**
- ✅ Bias 생성 성공
- 📊 α가 학습됨 (0.3~0.7 범위 탐색)
- 🎯 컴포넌트별 γ, σ 차이 보임

### 내일:
- 결과 분석
- 유망하면 → MSRAction3D 준비
- 문제 있으면 → 바이어스 강도 조정

---

## 📌 중요 판단 기준

### ✅ 성공 신호:
1. α가 0.4~0.6 범위 벗어남
2. Per-head σ, γ가 다양하게 학습됨
3. 컴포넌트 ablation에서 차이 보임

### ⚠️ 주의 신호:
1. α ≈ 0.5 고정 (WikiText처럼)
2. 모든 헤드 σ, γ 비슷함
3. 컴포넌트 제거해도 차이 없음

**→ 주의 신호 나오면:**
- w_align, w_sep, w_coh 2배 증가
- sigma_sep, sigma_coh 조정
- 재실험

---

**Status**: 🟢 Phase 1 완료, Phase 2 준비 완료
**Next**: 위 코드 실행 → 결과 분석 → 방향 결정
