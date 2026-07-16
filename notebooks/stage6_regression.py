# Stage 6: Negative binomial regression — RQ4 (associational co-occurrence of opening classes with helpfulness).

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

base = Path(__file__).resolve().parent.parent

df = pd.read_parquet(base / "07_taxonomy" / "final_classes.parquet")
print("shape:", df.shape)
print("columns:", df.columns.tolist())

df = df.rename(columns={"vote": "helpfulness_count",
                        "cluster_name": "cluster_label"})

df["word_count"] = df["reviewText"].str.split().str.len()
df["log_wc"]     = np.log1p(df["word_count"])
df["star_sq"]    = df["overall"] ** 2
df["reviewer_count"] = df.groupby("reviewerID")["reviewerID"].transform("count")
df["log_exp"]        = np.log1p(df["reviewer_count"])

mean_v = df["helpfulness_count"].mean()
var_v  = df["helpfulness_count"].var()
print(f"mean = {mean_v:.3f}")
print(f"var  = {var_v:.3f}")
print(f"variance / mean ratio = {var_v / mean_v:.3f}")

# ---- output folder ----
reg_dir = base / "09_regression"
reg_dir.mkdir(exist_ok=True)

# ---- Model 1: controls only ----
f1 = "helpfulness_count ~ log_wc + overall + star_sq + C(domain) + log_exp"
model1 = smf.negativebinomial(f1, data=df).fit(maxiter=1000, disp=0)
print(model1.summary())
print("Model 1 AIC:", round(model1.aic, 1))

# ---- Model 2: add opening class ----
f2 = f1 + ' + C(cluster_label, Treatment(reference="Direct Evaluation"))'
model2 = smf.negativebinomial(f2, data=df).fit(maxiter=1000, disp=0)
print(model2.summary())
print("Model 2 AIC:", round(model2.aic, 1))

# ---- Model 3: add opening class x domain interaction ----
f3 = f2 + ' + C(cluster_label, Treatment(reference="Direct Evaluation")):C(domain)'
model3 = smf.negativebinomial(f3, data=df).fit(maxiter=1000, disp=0)
print(model3.summary())
print("Model 3 AIC:", round(model3.aic, 1))

# ---- save coefficient tables ----
model1.summary2().tables[1].to_csv(reg_dir / "model1_summary.csv")
model2.summary2().tables[1].to_csv(reg_dir / "model2_summary.csv")
model3.summary2().tables[1].to_csv(reg_dir / "model3_summary.csv")
print("Saved model1, model2, and model3 summary tables")

# ---- AIC comparison ----
aic_table = pd.DataFrame({
    "model": ["M1: controls", "M2: + opening class", "M3: + class x domain"],
    "n_params": [len(model1.params), len(model2.params), len(model3.params)],
    "log_lik": [model1.llf, model2.llf, model3.llf],
    "AIC": [model1.aic, model2.aic, model3.aic],
})
aic_table["delta_AIC"] = aic_table["AIC"] - aic_table["AIC"].min()
aic_table.to_csv(reg_dir / "aic_comparison.csv", index=False)
print("\n", aic_table.to_string(index=False))
print("Saved aic_comparison.csv")

import matplotlib.pyplot as plt

# ---- IRR coefficient table (from Model 2) ----
params = model2.params
conf   = model2.conf_int()
irr = pd.DataFrame({
    "coef":     params,
    "IRR":      np.exp(params),
    "IRR_low":  np.exp(conf[0]),
    "IRR_high": np.exp(conf[1]),
    "p_value":  model2.pvalues,
})
irr = irr.drop(index="alpha").round(4)
irr.to_csv(reg_dir / "coefficient_table.csv")
print(irr.to_string())
print("Saved coefficient_table.csv")

# ---- coefficient plot (opening classes only) ----
class_rows = irr[irr.index.str.contains("cluster_label")].copy()
class_rows["label"] = [i.split("T.")[-1].rstrip("]") for i in class_rows.index]

fig, ax = plt.subplots(figsize=(8, 5))
y = range(len(class_rows))
ax.errorbar(class_rows["IRR"], y,
            xerr=[class_rows["IRR"] - class_rows["IRR_low"],
                  class_rows["IRR_high"] - class_rows["IRR"]],
            fmt="o", capsize=4, color="steelblue")
ax.axvline(x=1, color="red", linestyle="--", label="IRR = 1 (no difference)")
ax.set_yticks(list(y))
ax.set_yticklabels(class_rows["label"])
ax.set_xlabel("Incidence Rate Ratio (vs Direct Evaluation)")
ax.set_title("Opening class co-occurrence with helpfulness (Model 2)")
ax.legend()
plt.tight_layout()
fig.savefig(reg_dir / "coefficient_plot.png", dpi=150)
print("Saved coefficient_plot.png")