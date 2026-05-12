#Section 1: Imports
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt
from pathlib import Path

#Section 2: Load Embeddings + Index
base = Path(r"C:\Users\tpasumarthi\thesis_syntactic_openings")
embed_dir = base / "05_embeddings"
cluster_dir = base / "06_clusters"
cluster_dir.mkdir(exist_ok=True)

embeddings = np.load(embed_dir / "embeddings.npy")
index_df = pd.read_csv(embed_dir / "index.csv")

print(f"Embeddings shape: {embeddings.shape}")
print(f"Index rows:       {len(index_df)}")
assert embeddings.shape[0] == len(index_df), "Row mismatch!"
print("Sanity check passed — rows match.")

#Section 3: PCA dimentionality reduction
n_components = 50
pca = PCA(n_components=n_components, random_state=42)
X = pca.fit_transform(embeddings)
print(f"Reduced shape: {X.shape}")

cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"Variance explained by {n_components} components: {cumvar[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, n_components + 1), cumvar, marker="o", markersize=3)
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("PCA — Cumulative Explained Variance")
ax.axhline(y=0.80, color="red", linestyle="--", label="80% threshold")
ax.legend()
fig.tight_layout()
fig.savefig(cluster_dir / "pca_cumulative_variance.png", dpi=150)
plt.close()
print("Saved: pca_cumulative_variance.png")

#Section 4: KMeans sweep k=3 to 12
k_range = range(3, 13)
results = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    results.append({"k": k, "silhouette": sil, "davies_bouldin": db})
    print(f"k={k:2d}  silhouette={sil:.4f}  DB={db:.4f}")

results_df = pd.DataFrame(results)

#Section 5: Save evaluation results 
results_df.to_csv(cluster_dir / "evaluation_results.csv", index=False)
print("Saved: evaluation_results.csv")

#Section 6: Silhouette score plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(results_df["k"], results_df["silhouette"], marker="o")
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Silhouette score (higher is better)")
ax.set_title("Silhouette Score by k")
ax.set_xticks(list(k_range))
fig.tight_layout()
fig.savefig(cluster_dir / "silhouette_scores.png", dpi=150)
plt.close()
print("Saved: silhouette_scores.png")

#Section 7: Davies-Bouldin index plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(results_df["k"], results_df["davies_bouldin"], marker="o", color="orange")
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Davies-Bouldin index (lower is better)")
ax.set_title("Davies-Bouldin Index by k")
ax.set_xticks(list(k_range))
fig.tight_layout()
fig.savefig(cluster_dir / "davies_bouldin_scores.png", dpi=150)
plt.close()
print("Saved: davies_bouldin_scores.png")

#Section 8: Top candidates
print("\n=== Top candidates by silhouette (higher is better) ===")
print(results_df.nlargest(3, "silhouette").to_string(index=False))
print("\n=== Top candidates by Davies-Bouldin (lower is better) ===")
print(results_df.nsmallest(3, "davies_bouldin").to_string(index=False))

# Section 9: Stability analysis (ARI)
from sklearn.metrics import adjusted_rand_score
from itertools import combinations

candidate_ks = [5, 8, 10]
n_runs = 10
stability_results = []

for k in candidate_ks:
    all_labels = []
    for seed in range(n_runs):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        all_labels.append(labels)

    ari_scores = []
    for i, j in combinations(range(n_runs), 2):
        ari = adjusted_rand_score(all_labels[i], all_labels[j])
        ari_scores.append(ari)

    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    stability_results.append({"k": k, "mean_ari": mean_ari, "std_ari": std_ari})
    print(f"k={k:2d}  mean ARI={mean_ari:.4f}  std={std_ari:.4f}")

#Section 10: Stability summary
stability_df = pd.DataFrame(stability_results)
stability_df.to_csv(cluster_dir / "stability_results.csv", index=False)
print("\n=== Stability Summary ===")
print(stability_df.to_string(index=False))
print("\nSaved: stability_results.csv")

#Section 11: Decision log
log_dir = base / "08_Decision_Logs"
log_text = """# Decision Log — Number of Clusters (k)

## Decision
Final k = 8

## Candidates Evaluated
k = 3 to 12, evaluated with silhouette score and Davies-Bouldin index.
Top candidates (k=5, k=8, k=10) tested for stability via ARI across 10 runs.

## Evaluation Summary
| k  | Silhouette | Davies-Bouldin | Mean ARI | Std ARI |
|----|-----------|----------------|----------|---------|
| 5  | 0.1090    | 2.1651         | 0.9794   | 0.0143  |
| 8  | 0.1148    | 2.1279         | 0.9875   | 0.0066  |
| 10 | 0.1103    | 2.0929         | 0.9610   | 0.0375  |

## Rationale
k=8 selected because:
1. Highest mean ARI (0.9875) — most stable cluster assignments across runs
2. Lowest ARI std (0.0066) — stability is itself consistent
3. Competitive silhouette and Davies-Bouldin scores
4. Eight classes provide sufficient granularity for cross-domain taxonomy
   while remaining manageable for manual annotation (Stage 5)

## Pipeline Context
- PCA: 384 → 50 components (97.7% variance retained)
- KMeans: n_init=10, random_state=42 for final assignment
- Silhouette range across all k: 0.1055–0.1418 (typical for NLP clustering)
"""

with open(log_dir / "k_choice.md", "w", encoding="utf-8") as f:
    f.write(log_text)
print("Saved: k_choice.md")

#Section 12: Final KMeans with k=8
final_k = 8
km_final = KMeans(n_clusters=final_k, n_init=10, random_state=42)
final_labels = km_final.fit_predict(X)
print(f"Final clustering: k={final_k}")
print(f"Cluster sizes:")
for c in range(final_k):
    count = np.sum(final_labels == c)
    print(f"  Cluster {c}: {count} reviews ({count/len(final_labels)*100:.1f}%)")

#Section 13: Sample openings per cluster
templates_df = pd.read_parquet(base / "04_templates" / "openings_with_templates.parquet")
templates_df["cluster"] = final_labels

np.random.seed(42)
for c in range(final_k):
    cluster_rows = templates_df[templates_df["cluster"] == c]
    sample = cluster_rows.sample(n=25, random_state=42)
    print(f"\n{'='*60}")
    print(f"CLUSTER {c} — {len(cluster_rows)} reviews")
    print(f"{'='*60}")
    for _, row in sample.iterrows():
        print(f"  {row['open_10w']}")

#Section 14: Cluster name mapping + final outputs
cluster_names = {
    0: "Product Reference",
    1: "Acquisition Narrative",
    2: "Evaluative Positioning",
    3: "Direct Evaluation",
    4: "Specific Identification",
    5: "Affective Stance",
    6: "Contextual Framing",
    7: "Terse Evaluation"
}

templates_df["cluster_name"] = templates_df["cluster"].map(cluster_names)

taxonomy_dir = base / "07_taxonomy"
taxonomy_dir.mkdir(exist_ok=True)

templates_df.to_parquet(taxonomy_dir / "final_classes.parquet", index=False)
print(f"Saved: final_classes.parquet ({len(templates_df)} rows)")

taxonomy_rows = []
for c in range(final_k):
    count = np.sum(final_labels == c)
    taxonomy_rows.append({
        "cluster_id": c,
        "cluster_name": cluster_names[c],
        "count": count,
        "percentage": round(count / len(final_labels) * 100, 1)
    })

taxonomy_table = pd.DataFrame(taxonomy_rows)
taxonomy_table.to_csv(taxonomy_dir / "taxonomy_table.csv", index=False)
print("\nTaxonomy Table:")
print(taxonomy_table.to_string(index=False))
print("\nSaved: taxonomy_table.csv")
