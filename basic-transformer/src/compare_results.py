import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "logs/results_summary.csv"
OUT_DIR = "logs/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# 문자열 "True"/"False" → bool 변환
df["use_ascender"] = df["use_ascender"].astype(str).str.lower().map({"true": True, "false": False})

asc = df[(df["mode"] == "additive") & (df["use_ascender"] == True)]
base = df[(df["mode"] == "additive") & (df["use_ascender"] == False)]

if asc.empty or base.empty:
    print("⚠️ additive 모드에서 ascender=True/False 데이터가 둘 다 있는지 확인하세요.")
    print(df["use_ascender"].value_counts())
    raise SystemExit(0)

# === groupby 평균/표준편차 계산 ===
def summarize(group):
    return pd.DataFrame({
        "mean": [group["avg_loss"].mean()],
        "std": [group["avg_loss"].std()],
        "n": [len(group)]
    })

asc_agg = asc.groupby("epoch").apply(summarize).reset_index(drop=True)
base_agg = base.groupby("epoch").apply(summarize).reset_index(drop=True)

# 각 그룹의 epoch 추가
asc_agg["epoch"] = sorted(asc["epoch"].unique())
base_agg["epoch"] = sorted(base["epoch"].unique())

# === merge ===
comp = pd.merge(base_agg, asc_agg, on="epoch", suffixes=("_base", "_asc"))

# === 개선도 계산 ===
comp["diff"] = comp["mean_base"] - comp["mean_asc"]
comp["rel_impr_%"] = 100.0 * comp["diff"] / comp["mean_base"]

# === 결과 저장 ===
out_csv = os.path.join(OUT_DIR, "additive_vs_baseline.csv")
comp.to_csv(out_csv, index=False)
print(f"✅ Saved CSV: {out_csv}")

# === 시각화 ===
plt.figure(figsize=(7,5))
plt.plot(base_agg["epoch"], base_agg["mean"], label="Baseline (ASC off)", color="#3366cc")
plt.fill_between(base_agg["epoch"],
                 base_agg["mean"]-base_agg["std"],
                 base_agg["mean"]+base_agg["std"], alpha=0.2, color="#3366cc")

plt.plot(asc_agg["epoch"], asc_agg["mean"], label="ASCender (ASC on)", color="#cc3333")
plt.fill_between(asc_agg["epoch"],
                 asc_agg["mean"]-asc_agg["std"],
                 asc_agg["mean"]+asc_agg["std"], alpha=0.2, color="#cc3333")

plt.xlabel("Epoch")
plt.ylabel("Avg Loss")
plt.title("Additive Transformer: ASCender vs Baseline")
plt.legend()
plt.tight_layout()

out_img = os.path.join(OUT_DIR, "loss_curve_additive.png")
plt.savefig(out_img)
plt.close()
print(f"✅ Saved Plot: {out_img}")

# === 최종 결과 출력 ===
last = comp.iloc[-1]
print(f"\n[Final Epoch {int(last['epoch'])}]")
print(f"Baseline  mean±std: {last['mean_base']:.4f} ± {last['std_base']:.4f}")
print(f"ASCender  mean±std: {last['mean_asc']:.4f} ± {last['std_asc']:.4f}")
print(f"Δ (base-asc): {last['diff']:.4f}  |  Relative improvement: {last['rel_impr_%']:.2f}%")
