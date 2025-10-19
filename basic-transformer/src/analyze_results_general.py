import os
import json
import pandas as pd
import numpy as np


def analyze_results(csv_path="logs/results_summary.csv", save_json=True):
    if not os.path.exists(csv_path):
        print(f"❌ No CSV found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("❌ CSV is empty — no data to analyze.")
        return

    # ─────────────────────────────────────────────
    # 1️⃣ 마지막 epoch 기준으로 seed별 최종 loss 추출
    # ─────────────────────────────────────────────
    df_last = (
        df.groupby(["mode", "use_ascender", "bias_combo", "seed"])["avg_loss"]
        .last()
        .reset_index()
    )

    # ─────────────────────────────────────────────
    # 2️⃣ 각 bias 조합별 평균 및 표준편차 계산
    # ─────────────────────────────────────────────
    grouped = (
        df_last.groupby(["mode", "use_ascender", "bias_combo"])["avg_loss"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )

    # ─────────────────────────────────────────────
    # 3️⃣ 콘솔 요약 출력
    # ─────────────────────────────────────────────
    print("\n📊 ASCender Comparative Summary (Final Epoch)")
    print("──────────────────────────────────────────────────────────────")
    for _, row in grouped.iterrows():
        asc = "ASC" if row["use_ascender"] else "BASE"
        combo = row["bias_combo"]
        mean, std = row["mean"], row["std"]
        print(f"{asc:<5} | {combo:<10} | Loss = {mean:.4f} ± {std:.4f}")
    print("──────────────────────────────────────────────────────────────")

    # ─────────────────────────────────────────────
    # 4️⃣ Baseline vs Full ASCender 비교
    # ─────────────────────────────────────────────
    base_loss = grouped.query("use_ascender == False")["mean"].mean()
    full_loss = (
        grouped.query("bias_combo == 'A+S+C'")["mean"].mean()
        if "A+S+C" in grouped["bias_combo"].values
        else None
    )

    if full_loss is not None:
        delta = full_loss - base_loss
        print(
            f"\nΔLoss (Full - Baseline) = {delta:+.4f} → "
            f"{'Improved ✅' if delta < 0 else 'Worse ⚠️'}"
        )

    # ─────────────────────────────────────────────
    # 5️⃣ 추가 분석: 각 bias 조합별 상대 성능
    # ─────────────────────────────────────────────
    if full_loss is not None:
        print("\n📈 Relative Performance (vs Baseline)")
        print("────────────────────────────────────")
        for _, row in grouped.iterrows():
            if not row["use_ascender"]:
                continue
            delta = row["mean"] - base_loss
            trend = "✅" if delta < 0 else "⚠️"
            print(f"{row['bias_combo']:<10}  ΔLoss={delta:+.4f}  {trend}")
        print("────────────────────────────────────")

    # ─────────────────────────────────────────────
    # 6️⃣ 결과 JSON 저장
    # ─────────────────────────────────────────────
    if save_json:
        save_dir = os.path.dirname(csv_path)
        save_path = os.path.join(save_dir, "compare_summary.json")
        grouped.to_json(save_path, orient="records", indent=2)
        print(f"\n💾 Saved summary → {save_path}")

    print("\n✅ Analysis complete.\n")


if __name__ == "__main__":
    analyze_results()
