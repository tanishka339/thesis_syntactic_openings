# Decision Log — Number of Clusters (k)

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
