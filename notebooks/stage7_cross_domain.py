# stage7_cross_domain.py
# Stage 7 — Cross-domain analysis (Books vs Electronics)
#   Q1 composition : does the class mix differ by domain?          -> chi-square
#   Q2 association  : does helpfulness within a class differ by domain? -> Kruskal-Wallis
# THIS CHUNK: locate root, load data, and VERIFY real column names + domain
# labels before writing any formula that assumes them.

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# Path-safety rule: never hard-code D:\...  Derive the root from THIS file.
# The script lives in <root>/notebooks/, so parent.parent == project root.
base = Path(__file__).resolve().parent.parent

# The one file Stage 7 needs — it already carries every column (no joins).
data_path = base / "07_taxonomy" / "final_classes.parquet"

# Where today's outputs will live. Create it now if it isn't there yet.
out_dir = base / "07_taxonomy" / "domain_comparison"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(data_path)

# --- VERIFY REALITY before trusting any column name ---
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nDomain:\n", df["domain"].value_counts(dropna=False))
print("\nClass id -> name (with counts):\n",
      df.groupby(["cluster", "cluster_name"]).size())
print("\nHelpfulness (vote):\n", df["vote"].describe())

# Q1 COMPOSITION: does the class MIX differ between Books and Electronics?
# (This is about proportions of classes, NOT about helpfulness.)

# Contingency table: rows = opening class, cols = domain, cells = review counts.
observed = pd.crosstab(df["cluster_name"], df["domain"])
print("Observed counts (class x domain):\n", observed, "\n")

# chi2_contingency returns FOUR values in this fixed order:
#   chi2     : the test statistic
#   p        : its p-value
#   dof      : degrees of freedom = (rows-1)*(cols-1) = 7*1 = 7
#   expected : counts we'd expect if class and domain were independent
chi2, p, dof, expected = chi2_contingency(observed)

# ASSUMPTION CHECK: chi-square is trustworthy only if every expected cell >= 5.
print("Expected counts:\n",
      pd.DataFrame(expected, index=observed.index, columns=observed.columns), "\n")
print(f"Smallest expected cell: {expected.min():.1f}  (need >= 5)\n")

# Cramer's V: chi2 says IF distributions differ; V says HOW MUCH (0 to 1).
# V = sqrt( chi2 / (n * min(rows-1, cols-1)) ).  Here cols-1 = 1.
n = int(observed.to_numpy().sum())
min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Chi-square = {chi2:.2f}, dof = {dof}, p = {p:.4g}")
print(f"Cramer's V = {cramers_v:.4f}")

# Save the contingency table now — it's the basis for Table 9.
observed.to_csv(out_dir / "class_frequency_by_domain.csv", encoding="utf-8")

# Q2 ASSOCIATION: WITHIN each class, does helpfulness differ by domain?
# Kruskal-Wallis on 'vote', grouped by 'domain', run once per class.
# NOTE: with exactly 2 domains, Kruskal-Wallis is equivalent to Mann-Whitney U.
# This is DESCRIPTIVE confirmation of Stage 6's Model 3 interactions, NOT a
# re-fit: KW is marginal/unadjusted, the regression interaction was covariate-
# adjusted. We report convergence between two methods, not a second model.

rows = []
for cid in range(8):                          # 0..7 = stable taxonomy order
    name = df.loc[df["cluster"] == cid, "cluster_name"].iloc[0]
    sub  = df[df["cluster"] == cid]           # this class only

    books = sub.loc[sub["domain"] == "Books", "vote"]
    elec  = sub.loc[sub["domain"] == "Electronics", "vote"]

    H, p  = kruskal(books, elec)              # H statistic, raw p-value
    n     = len(sub)
    eps_sq = H / (n - 1)                       # epsilon-squared effect size

    rows.append({
        "cluster": cid, "cluster_name": name, "n": n,
        "median_books": books.median(), "median_elec": elec.median(),
        "mean_books": round(books.mean(), 2), "mean_elec": round(elec.mean(), 2),
        "H": round(H, 3), "p_raw": p, "epsilon_sq": round(eps_sq, 5),
    })

res = pd.DataFrame(rows)

# Benjamini-Hochberg FDR across the 8 tests: 8 unadjusted tests at alpha=.05
# would inflate the family-wise false-positive rate; BH controls it.
reject, p_fdr, _, _ = multipletests(res["p_raw"], alpha=0.05, method="fdr_bh")
res["p_fdr"]   = p_fdr
res["sig_fdr"] = reject

# Direction (by mean, robust-ish given the skew): which domain reads higher.
res["higher"] = np.where(res["mean_elec"] > res["mean_books"], "Electronics", "Books")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print(res.sort_values("cluster").to_string(index=False))

res.to_csv(out_dir / "kruskal_by_class.csv", index=False, encoding="utf-8")

# FR12 FOLLOW-UP (descriptive only, NOT a re-fit):
# Full helpfulness distribution by domain for the two Stage 6 interaction classes.
# We report these to show the marginal picture CONVERGES with Model 3's signs.
#   Acquisition Narrative: Model 3 interaction +0.18 -> expect higher in Electronics
#   Terse Evaluation      : Model 3 interaction -0.35 -> expect higher in Books
flagged = {"Acquisition Narrative": +0.18, "Terse Evaluation": -0.35}

for name, coef in flagged.items():
    sub = df[df["cluster_name"] == name]
    print(f"\n=== {name}  (Stage 6 class x Electronics coef = {coef:+.2f}) ===")
    # describe() per domain: count, mean, std, quartiles side by side.
    print(sub.groupby("domain")["vote"].describe().round(2))
    dirn = "Electronics" if coef > 0 else "Books"
    print(f"Model 3 predicts higher helpfulness in: {dirn}")

# --- PLOT 1: opening-class frequency by domain (grouped bars) ---
# Visualizes the chi-square result: same 8 classes in both domains,
# with the small compositional tilts. Counts are fair here because the
# two domains are near-balanced (4999 vs 5000).

# Consistent class order = taxonomy id order 0..7, NOT alphabetical,
# so every figure in the thesis lists the classes in the same sequence.
order = (df.drop_duplicates("cluster")
           .sort_values("cluster")["cluster_name"].tolist())

freq = observed.reindex(order)          # crosstab rows, re-sorted to id order
x = np.arange(len(order))               # one x-slot per class: [0,1,...,7]
w = 0.4                                  # bar width; two 0.4 bars fit one slot

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - w/2, freq["Books"],       width=w, label="Books")
ax.bar(x + w/2, freq["Electronics"], width=w, label="Electronics")

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("Number of reviews")
ax.set_title("Opening-class frequency by domain (Books vs Electronics)")
ax.legend(title="Domain")
fig.tight_layout()                       # stop the rotated labels being clipped

fig.savefig(out_dir / "class_frequency_grouped_bar.png", dpi=200)
plt.close(fig)
print("Saved class_frequency_grouped_bar.png")

# --- PLOT 2: helpfulness by class and domain (grouped bars, MEDIANS) ---
# We plot MEDIAN vote, not mean: 'vote' is heavily right-skewed (e.g. Acq.
# Narrative Electronics mean 14.86 but median 5.0, max 643). Medians describe
# the TYPICAL review and are robust to the viral-review outliers; means would
# make the bars a story about outliers. This choice is stated in the y-label.

med = res.set_index("cluster_name").reindex(order)   # same class order as Plot 1

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - w/2, med["median_books"], width=w, label="Books")
ax.bar(x + w/2, med["median_elec"],  width=w, label="Electronics")

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("Median helpfulness votes")
ax.set_title("Median helpfulness by opening class and domain")
ax.legend(title="Domain")
fig.tight_layout()

fig.savefig(out_dir / "helpfulness_by_class_and_domain.png", dpi=200)
plt.close(fig)
print("Saved helpfulness_by_class_and_domain.png")