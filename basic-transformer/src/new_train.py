# train.py
from __future__ import annotations
import math
import argparse
import random
import time
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.transformer import (
    Transformer, TransformerConfig, LabelSmoothingLoss, NoamLR
)

# =========================================================
# Utils: seeding, schedules, metric helpers
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def cosine_tau(step: int, total_steps: int, start_tau: float, end_tau: float) -> float:
    """ASC-head 전용 softmax 온도 τ 코사인 스케줄."""
    if total_steps <= 0:
        return start_tau
    t = min(step / total_steps, 1.0)
    # cosine decay: start -> end
    return end_tau + 0.5 * (start_tau - end_tau) * (1 + math.cos(math.pi * t))

def apply_tau_to_asc_heads(model: Transformer, tau: float):
    """decoder self-attn L0/L1에만 τ 적용 (존재 시)."""
    if hasattr(model, "decoder"):
        layers = getattr(model.decoder, "layers", [])
        for li in (0, 1):
            if li < len(layers):
                mha = layers[li].self_attn
                if hasattr(mha, "attn_temperature"):
                    mha.attn_temperature = float(tau)

def _safe_softmax(masked_logits: torch.Tensor) -> torch.Tensor:
    # masked_logits: (B,H,T,S) w/ -inf on masked positions (already)
    # clamp for numerical stability then softmax on last dim
    x = torch.nan_to_num(masked_logits, nan=-1e9, posinf=80.0, neginf=-80.0).clamp(-80, 80)
    return torch.softmax(x, dim=-1)

@torch.no_grad()
def collect_attn_metrics(model: Transformer) -> Dict[str, float]:
    """
    transformer.py의 MHA가 저장해 둔 스냅샷을 이용해
    Δp(= |softmax(post)-softmax(pre)|_1 평균)와 KL(post||pre)을 추정.
    decoder self L0/L1만 집계(존재하는 것만).
    """
    deltas: List[float] = []
    kls: List[float] = []

    if not hasattr(model, "decoder"):  # 방어
        return {"delta_p_mean": 0.0, "kl_mean": 0.0}

    for li in (0, 1):
        if li >= len(model.decoder.layers):
            continue
        mha = model.decoder.layers[li].self_attn
        pre = getattr(mha, "attn_pre_masked", None)   # (B,H,T,S) with -inf on masked
        post = getattr(mha, "attn_post_masked", None) # (B,H,T,S) same mask
        if pre is None or post is None:
            continue

        p0 = _safe_softmax(pre.float())
        p1 = _safe_softmax(post.float())
        # Δp = mean L1 distance
        delta = torch.mean(torch.abs(p1 - p0)).item()
        deltas.append(delta)

        # KL(post || pre) with small eps
        eps = 1e-8
        kl = torch.mean((p1 + eps) * (torch.log(p1 + eps) - torch.log(p0 + eps))).item()
        kls.append(kl)

    if len(deltas) == 0:
        return {"delta_p_mean": 0.0, "kl_mean": 0.0}
    return {
        "delta_p_mean": float(sum(deltas) / len(deltas)),
        "kl_mean": float(sum(kls) / len(kls))
    }

def ppl_from_loss(loss_val: float) -> float:
    try:
        return float(math.exp(loss_val))
    except OverflowError:
        return float("inf")

# =========================================================
# Core training/eval
# =========================================================

@dataclass
class TrainArgs:
    epochs: int = 3
    smoothing: float = 0.05
    warmup: int = 4000
    clip_grad: float = 1.0
    lr: float = 1.0  # Noam과 함께 쓰는 스칼라
    asc_use: bool = True
    tau_start: float = 0.95
    tau_end: float = 0.90
    # 총 스텝수는 dataloader 크기 * epochs로 동적으로 계산

def build_model(vocab_src: int, vocab_tgt: int, cfg_overrides: Dict[str, Any] = None) -> Transformer:
    cfg = TransformerConfig(
        src_vocab_size=vocab_src,
        tgt_vocab_size=vocab_tgt,
        d_model=256,
        n_heads=8,
        n_layers_enc=3,
        n_layers_dec=3,
        d_ff=1024,
        dropout=0.1,
        pad_id=0,
        tie_embeddings=True,
        use_ascender=cfg_overrides.get("use_ascender", False) if cfg_overrides else False,
    )
    if cfg_overrides:
        # 선택적 오버라이드
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    model = Transformer(cfg)
    return model

def run_epoch(
    model: Transformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch_idx: int,
    total_steps: int,
    args: TrainArgs,
    train: bool = True,
) -> Dict[str, float]:
    model.train(train)
    running_loss = 0.0
    running_tokens = 0
    steps_done = 0

    # 스텝 인덱스는 스케줄을 위해 누적 관리
    if not hasattr(model, "_global_step"):
        model._global_step = 0

    t0 = time.time()
    for batch in loader:
        # 기대: batch = (src, tgt_inp, tgt_out)
        src, tgt_inp, tgt_out = batch
        src = src.to(device, non_blocking=True)
        tgt_inp = tgt_inp.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)

        # ASC-head 전용 τ 스케줄 적용
        if args.asc_use and model.cfg.use_ascender:
            tau = cosine_tau(model._global_step, total_steps, args.tau_start, args.tau_end)
            apply_tau_to_asc_heads(model, tau)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(src, tgt_inp)  # (B,T,V)
        loss = criterion(logits, tgt_out)

        if train:
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()

        # 통계
        B, T, V = logits.shape
        running_loss += float(loss.detach().item()) * (B * T)
        running_tokens += (B * T)

        model._global_step += 1
        steps_done += 1

    avg_loss = running_loss / max(running_tokens, 1)
    metrics = {
        "loss": avg_loss,
        "ppl": ppl_from_loss(avg_loss),
        "steps": steps_done,
        "tok_per_sec": running_tokens / max(1e-6, (time.time() - t0)),
    }

    # Δp/KL 수집 (eval일 때도 가능)
    try:
        m = collect_attn_metrics(model)
        metrics.update(m)
    except Exception:
        # 안전장치: 수집 실패 시 조용히 진행
        pass

    return metrics

@torch.no_grad()
def evaluate(model: Transformer, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_tokens = 0
    for batch in loader:
        src, tgt_inp, tgt_out = batch
        src = src.to(device, non_blocking=True)
        tgt_inp = tgt_inp.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)

        logits = model(src, tgt_inp)
        loss = criterion(logits, tgt_out)

        B, T, _ = logits.shape
        running_loss += float(loss.detach().item()) * (B * T)
        running_tokens += (B * T)

    avg = running_loss / max(running_tokens, 1)
    return {"val_loss": avg, "val_ppl": ppl_from_loss(avg)}

# =========================================================
# A/B runner (3-seed 평균)
# =========================================================

def run_ab_3seeds(
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_src: int,
    vocab_tgt: int,
    device: torch.device,
    args: TrainArgs,
    cfg_overrides: Dict[str, Any] | None = None,
    seeds: Tuple[int, int, int] = (42, 43, 44),
):
    results = {"A(base_off)": [], "B(asc_on)": []}

    for mode_name, use_asc in [("A(base_off)", False), ("B(asc_on)", True)]:
        print(f"\n=== [{mode_name}] use_ascender={use_asc} ===")
        for sd in seeds:
            set_seed(sd)
            # 모델 구성
            ov = dict(cfg_overrides or {})
            ov["use_ascender"] = bool(use_asc)
            model = build_model(vocab_src, vocab_tgt, ov).to(device)

            # 옵티마/스케줄/로스
            crit = LabelSmoothingLoss(vocab_tgt, smoothing=args.smoothing, ignore_index=0).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
            sched = NoamLR(opt, d_model=model.cfg.d_model, warmup_steps=args.warmup)

            # 전체 스텝수(τ 스케줄용)
            total_steps = len(train_loader) * args.epochs

            log_hist = []
            for ep in range(1, args.epochs + 1):
                tr = run_epoch(model, train_loader, opt, sched, crit, device, ep, total_steps, args, train=True)
                ev = evaluate(model, val_loader, crit, device)
                merged = dict(seed=sd, epoch=ep, **tr, **ev)
                log_hist.append(merged)
                print(f"[{mode_name}|seed={sd}|ep={ep}] "
                      f"loss={tr['loss']:.4f} ppl={tr['ppl']:.2f} "
                      f"Δp={tr.get('delta_p_mean', 0.0):.4f} KL={tr.get('kl_mean', 0.0):.4f} "
                      f"| val_loss={ev['val_loss']:.4f} val_ppl={ev['val_ppl']:.2f}")

            # 마지막 에폭의 검증지표를 대표값으로 사용
            results[mode_name].append(log_hist[-1])

    def summarize(key: str) -> Tuple[float, float]:
        vals = [r[key] for r in results["B(asc_on)"]]  # ASC
        base = [r[key.replace("val_", "val_")] for r in results["A(base_off)"]]
        mean_asc = sum(vals) / len(vals)
        mean_base = sum(base) / len(base)
        return mean_base, mean_asc

    # 리포트
    mb_loss = sum(r["val_loss"] for r in results["A(base_off)"]) / len(results["A(base_off)"])
    ma_loss = sum(r["val_loss"] for r in results["B(asc_on)"]) / len(results["B(asc_on)"])
    mb_ppl  = sum(r["val_ppl"]  for r in results["A(base_off)"]) / len(results["A(base_off)"])
    ma_ppl  = sum(r["val_ppl"]  for r in results["B(asc_on)"]) / len(results["B(asc_on)"])

    print("\n📊 A/B (3-seeds mean, last-epoch, validation):")
    print(f"Baseline   val_loss={mb_loss:.4f}  val_ppl={mb_ppl:.2f}")
    print(f"ASCender   val_loss={ma_loss:.4f}  val_ppl={ma_ppl:.2f}")
    print(f"Δ (base-asc): {mb_loss - ma_loss:+.4f}  |  Relative improvement: {100.0*(mb_loss - ma_loss)/max(1e-9,mb_loss):+.2f}%")

# =========================================================
# Main (예시)
# =========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--tau_start", type=float, default=0.95)
    p.add_argument("--tau_end", type=float, default=0.90)
    p.add_argument("--asc_use", action="store_true", help="단건 학습(run_epoch) 모드에서 ASC 사용")
    p.add_argument("--ab", action="store_true", help="3-seed A/B 비교 러너 실행")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def get_loaders() -> Tuple[DataLoader, DataLoader, int, int]:
    """
    ⚠️ 프로젝트 환경에 맞게 교체하세요.
    DataLoader는 (src, tgt_inp, tgt_out) 튜플을 배치로 반환해야 합니다.
    pad_id=0 가정.
    """
    raise NotImplementedError("프로젝트의 데이터로더를 연결하세요. (src,tgt_inp,tgt_out) 배치를 반환해야 합니다.")

def main():
    args_ns = parse_args()
    device = torch.device(args_ns.device)

    # Data
    train_loader, val_loader, vocab_src, vocab_tgt = get_loaders()

    train_args = TrainArgs(
        epochs=args_ns.epochs,
        smoothing=args_ns.smoothing,
        warmup=args_ns.warmup,
        clip_grad=args_ns.clip_grad,
        lr=args_ns.lr,
        asc_use=args_ns.asc_use,
        tau_start=args_ns.tau_start,
        tau_end=args_ns.tau_end,
    )

    if args_ns.ab:
        # 3-seed A/B 자동 비교
        run_ab_3seeds(
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            device=device,
            args=train_args,
            cfg_overrides={},  # 필요 시 TransformerConfig override 가능
            seeds=(42, 43, 44),
        )
        return

    # 단건 학습(ASC on/off는 --asc_use로 제어)
    set_seed(42)
    model = build_model(vocab_src, vocab_tgt, {"use_ascender": bool(args_ns.asc_use)}).to(device)
    crit = LabelSmoothingLoss(vocab_tgt, smoothing=train_args.smoothing, ignore_index=0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_args.lr, betas=(0.9, 0.98), eps=1e-9)
    sched = NoamLR(opt, d_model=model.cfg.d_model, warmup_steps=train_args.warmup)

    total_steps = len(train_loader) * train_args.epochs
    for ep in range(1, train_args.epochs + 1):
        tr = run_epoch(model, train_loader, opt, sched, crit, device, ep, total_steps, train_args, train=True)
        ev = evaluate(model, val_loader, crit, device)
        print(f"[single|ep={ep}] loss={tr['loss']:.4f} ppl={tr['ppl']:.2f} "
              f"Δp={tr.get('delta_p_mean', 0.0):.4f} KL={tr.get('kl_mean', 0.0):.4f} "
              f"| val_loss={ev['val_loss']:.4f} val_ppl={ev['val_ppl']:.2f}")

if __name__ == "__main__":
    main()
