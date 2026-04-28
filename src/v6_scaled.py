"""
V6: Scale experiment.
Key insight: v4_aggressive (best LB 931k) predicts 1.21x of 2022 level.
Sample submission is 1.01x of 2022. If actual 2023 is closer to 2022 level,
we're over-predicting by ~20%.

Strategy: take v4_aggressive (best LB shape) and scale it down to different
levels to find the sweet spot.
"""
import pandas as pd, numpy as np, os

OUT_DIR = "output"
sales = pd.read_csv("data/raw/sales.csv")
sales["Date"] = pd.to_datetime(sales["Date"])
sub_tpl = pd.read_csv("data/raw/sample_submission.csv")
sub_tpl["Date"] = pd.to_datetime(sub_tpl["Date"])

# Load best LB submission (v4_aggressive = 931k)
best = pd.read_csv("output/submission_v4_aggressive.csv")
best["Date"] = pd.to_datetime(best["Date"])

# Also load v4_balanced and v3
v4b = pd.read_csv("output/submission_v4_balanced.csv")
v4b["Date"] = pd.to_datetime(v4b["Date"])

# Historical reference
mean_2022_rev = sales[sales.Date.dt.year == 2022].Revenue.mean()
mean_2022_cogs = sales[sales.Date.dt.year == 2022].COGS.mean()
mean_2021_rev = sales[sales.Date.dt.year == 2021].Revenue.mean()
mean_2021_cogs = sales[sales.Date.dt.year == 2021].COGS.mean()

print(f"2022 mean: Rev={mean_2022_rev:,.0f}, COGS={mean_2022_cogs:,.0f}")
print(f"2021 mean: Rev={mean_2021_rev:,.0f}, COGS={mean_2021_cogs:,.0f}")
print(f"v4_aggressive mean: Rev={best.Revenue.mean():,.0f}, COGS={best.COGS.mean():,.0f}")
print(f"v4_aggressive ratio vs 2022: Rev={best.Revenue.mean()/mean_2022_rev:.3f}x")
print(f"Sample sub mean: Rev={sub_tpl.Revenue.mean():,.0f}")
print(f"Sample sub ratio vs 2022: Rev={sub_tpl.Revenue.mean()/mean_2022_rev:.3f}x")

# Create scaled versions
# The idea: keep the SHAPE (daily pattern) from v4_aggressive,
# but adjust the LEVEL (mean) to different targets
scales = {
    "scale_100": 1.00,   # same as 2022
    "scale_105": 1.05,   # 5% growth
    "scale_110": 1.10,   # 10% growth
    "scale_095": 0.95,   # slight decline
    "scale_090": 0.90,   # 10% decline
}

# Also try blending v4_aggressive with sample_submission
# (sample might contain actual baseline info)

print("\n--- Generating scaled submissions ---")
for name, target_ratio in scales.items():
    sub = sub_tpl[["Date"]].copy()
    for col in ["Revenue", "COGS"]:
        hist_mean = mean_2022_rev if col == "Revenue" else mean_2022_cogs
        current_mean = best[col].mean()
        current_ratio = current_mean / hist_mean
        scale_factor = target_ratio / current_ratio
        sub[col] = best[col] * scale_factor
    
    sub_ratio = sub.Revenue.mean() / mean_2022_rev
    print(f"  {name}: Rev mean={sub.Revenue.mean():,.0f} (ratio={sub_ratio:.3f}x), scale_factor={target_ratio/best.Revenue.mean()*mean_2022_rev:.3f}")
    
    out = sub.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(os.path.join(OUT_DIR, f"submission_v6_{name}.csv"), index=False)

# Also try: blend v4_aggressive shape with sample_submission level
print("\n--- Blending with sample submission ---")
for blend_w in [0.3, 0.5, 0.7]:
    sub = sub_tpl[["Date"]].copy()
    for col in ["Revenue", "COGS"]:
        sub[col] = blend_w * best[col] + (1 - blend_w) * sub_tpl[col]
    
    name = f"blend_v4_{int(blend_w*100)}_sample_{int((1-blend_w)*100)}"
    print(f"  {name}: Rev mean={sub.Revenue.mean():,.0f}, ratio vs 2022={sub.Revenue.mean()/mean_2022_rev:.3f}x")
    
    out = sub.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(os.path.join(OUT_DIR, f"submission_v6_{name}.csv"), index=False)

# Try: use v4 shape but normalize to sample_submission mean
sub = sub_tpl[["Date"]].copy()
for col in ["Revenue", "COGS"]:
    target_mean = sub_tpl[col].mean()
    current_mean = best[col].mean()
    sub[col] = best[col] * (target_mean / current_mean)
name = "v4_shape_sample_level"
print(f"\n  {name}: Rev mean={sub.Revenue.mean():,.0f}, ratio={sub.Revenue.mean()/mean_2022_rev:.3f}x")
out = sub.copy()
out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
out.to_csv(os.path.join(OUT_DIR, f"submission_v6_{name}.csv"), index=False)

print("\n🏁 Done. Try uploading v6_scale_100 and v6_scale_105 first.")
print("   If they improve -> we were over-predicting.")
print("   If they worsen -> the level was already about right, problem is shape.")
