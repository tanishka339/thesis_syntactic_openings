"""
Stage 4 — Cluster Output Report
===============================

Produces the two deliverables requested by Prof. Mora:

  (1) A set of 10 reviews belonging to each cluster.
  (2) Basic statistics for the total sample and then for each cluster,
      plus distribution plots in the same style as Stage 1, section
      "3. Distribution Plots" (star rating, word count, helpfulness votes).

This script does NOT re-run clustering. It reads the artefacts that
stage4_cluster.py already wrote and reports on them, so the modelling
stage and the reporting stage stay separate and independently re-runnable.

USAGE
-----
    python stage4_output.py --inspect     # print columns of both files, then exit
    python stage4_output.py               # generate all outputs

If the default paths below are wrong, either edit CONFIG or pass them in:

    python stage4_output.py --corpus path/to/corpus.parquet \
                            --taxonomy path/to/final_classes.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edit these if you don't want to pass paths on the command line
# ---------------------------------------------------------------------------

CONFIG = {
    "corpus_path":   Path("01_processed/corpus.parquet"),
    "taxonomy_path": Path("07_taxonomy/final_classes.parquet"),
    "stats_out":     Path("07_taxonomy"),
    "plots_out":     Path("06_clusters"),
    "n_samples":     10,
    "seed":          42,
    "wordcount_cap": 500,   # matches Stage 1 Figure 1, centre panel
    "votes_cap":     50,    # matches Stage 1 Figure 1, right panel
}

# Candidate column names, in priority order. The script picks the first that
# exists, case-insensitively, so it survives small naming differences between
# your stages without you having to hunt through the code.
CANDIDATES = {
    "text":     ["reviewText", "review_text", "text", "body"],
    "opening":  ["opening", "opening_text", "opening_5w", "opening_10w",
                 "opening_segment", "first_sentence"],
    "rating":   ["overall", "star_rating", "stars", "rating"],
    "votes":    ["helpfulness_count", "helpfulness_votes", "helpful_votes",
                 "helpful_count", "helpful", "votes", "vote"],
    "words":    ["word_count", "n_words", "length_words"],
    "domain":   ["domain", "category", "product_domain"],
    "cluster":  ["cluster", "cluster_id", "label", "class_id", "final_class"],
    "clustnm":  ["cluster_name", "class_name", "label_name", "taxonomy_name"],
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def find_col(df: pd.DataFrame, role: str, required: bool = True):
    """Return the actual column name in df matching a logical role."""
    lookup = {c.lower(): c for c in df.columns}
    for cand in CANDIDATES[role]:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    if required:
        raise KeyError(
            f"Could not find a '{role}' column. Looked for "
            f"{CANDIDATES[role]}. Available columns: {list(df.columns)}"
        )
    return None


def coerce_votes(series: pd.Series) -> pd.Series:
    """
    Amazon dumps sometimes store helpfulness as [helpful, total] rather than a
    scalar. Take the first element in that case, then force numeric.
    """
    if series.dtype == object:
        first = series.dropna().iloc[0] if series.notna().any() else None
        if isinstance(first, (list, tuple, np.ndarray)):
            series = series.apply(
                lambda v: v[0] if isinstance(v, (list, tuple, np.ndarray)) and len(v) else np.nan
            )
    return pd.to_numeric(series, errors="coerce")


def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load(corpus_path: Path, taxonomy_path: Path):
    for p in (corpus_path, taxonomy_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} does not exist. Run with --inspect after fixing the path, "
                f"or pass --corpus / --taxonomy explicitly."
            )
    corpus = pd.read_parquet(corpus_path)
    taxonomy = pd.read_parquet(taxonomy_path)
    print(f"corpus   : {corpus.shape[0]:,} rows x {corpus.shape[1]} cols  <- {corpus_path}")
    print(f"taxonomy : {taxonomy.shape[0]:,} rows x {taxonomy.shape[1]} cols  <- {taxonomy_path}")
    return corpus, taxonomy


def inspect(corpus_path: Path, taxonomy_path: Path) -> None:
    """Print structure of both files so you can confirm column names."""
    corpus, taxonomy = load(corpus_path, taxonomy_path)
    for name, df in (("CORPUS", corpus), ("TAXONOMY", taxonomy)):
        banner(f"{name}: columns and dtypes")
        print(df.dtypes.to_string())
        banner(f"{name}: first 3 rows")
        print(df.head(3).to_string(max_colwidth=50, line_width=200))


# ---------------------------------------------------------------------------
# 2. Merge — on a shared key if one exists, positionally only as a fallback
# ---------------------------------------------------------------------------

def merge(corpus: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    cl = find_col(taxonomy, "cluster")
    nm = find_col(taxonomy, "clustnm", required=False)

    keep = [cl] + ([nm] if nm else [])

    # Prefer a genuine join key so row order can never silently mislabel reviews.
    shared_ids = [
        c for c in corpus.columns
        if c in taxonomy.columns
        and ("id" in c.lower() or c.lower() in {"asin", "reviewerid", "review_id"})
    ]

    if shared_ids:
        key = shared_ids[0]
        if corpus[key].is_unique and taxonomy[key].is_unique:
            print(f"merging on shared key '{key}'")
            out = corpus.merge(taxonomy[[key] + keep], on=key,
                               how="inner", validate="1:1")
            if len(out) != len(corpus):
                print(f"  note: {len(corpus) - len(out):,} corpus rows had no cluster label")
            return out
        print(f"shared key '{key}' is not unique — falling back to row order")

    if len(corpus) != len(taxonomy):
        raise ValueError(
            f"No usable join key, and the row counts differ "
            f"({len(corpus):,} vs {len(taxonomy):,}). These files cannot be "
            f"aligned safely. Re-export final_classes.parquet with an id column."
        )

    print("no shared id column — aligning by row order (counts match)")
    return pd.concat(
        [corpus.reset_index(drop=True), taxonomy[keep].reset_index(drop=True)],
        axis=1,
    )


# ---------------------------------------------------------------------------
# 3. Normalise the columns we report on
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    text = find_col(df, "text")
    cols = {
        "text":    text,
        "opening": find_col(df, "opening", required=False),
        "rating":  find_col(df, "rating"),
        "votes":   find_col(df, "votes"),
        "domain":  find_col(df, "domain", required=False),
        "cluster": find_col(df, "cluster"),
        "clustnm": find_col(df, "clustnm", required=False),
    }

    df = df.copy()

    wc = find_col(df, "words", required=False)
    if wc is None:
        wc = "word_count"
        df[wc] = df[cols["text"]].fillna("").str.split().str.len()
        print("word_count not found — computed from review text")
    cols["words"] = wc

    df[cols["votes"]] = coerce_votes(df[cols["votes"]])
    df[cols["rating"]] = pd.to_numeric(df[cols["rating"]], errors="coerce")

    # If the taxonomy carried no names, fall back to "Cluster 0", "Cluster 1", ...
    if cols["clustnm"] is None:
        cols["clustnm"] = "cluster_name"
        df[cols["clustnm"]] = "Cluster " + df[cols["cluster"]].astype(str)
        print("cluster_name not found — using generic Cluster N labels")

    dropped = df[[cols["rating"], cols["votes"], cols["words"]]].isna().any(axis=1).sum()
    if dropped:
        print(f"warning: {dropped:,} rows have a missing rating/votes/word_count value")

    print(f"\n{len(df):,} reviews across {df[cols['cluster']].nunique()} clusters")
    return df, cols


# ---------------------------------------------------------------------------
# 4. Deliverable 1 — 10 reviews per cluster
# ---------------------------------------------------------------------------

def sample_reviews(df, cols, out_dir: Path, n: int, seed: int) -> pd.DataFrame:
    rows = []
    for cid, grp in df.groupby(cols["cluster"], sort=True):
        take = grp.sample(n=min(n, len(grp)), random_state=seed)
        rows.append(take)
    samp = pd.concat(rows)

    export = [cols["cluster"], cols["clustnm"]]
    if cols["opening"]:
        export.append(cols["opening"])
    export += [cols["text"], cols["rating"], cols["votes"], cols["words"]]
    if cols["domain"]:
        export.append(cols["domain"])

    samp = samp[export].rename(columns={
        cols["cluster"]: "cluster", cols["clustnm"]: "cluster_name",
        cols["rating"]: "star_rating", cols["votes"]: "helpfulness_votes",
        cols["words"]: "word_count",
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sample_reviews_per_cluster.csv"
    samp.to_csv(csv_path, index=False, encoding="utf-8")

    # A readable companion file — a CSV of long review text is painful to skim,
    # so we also write a Markdown version with the first 600 characters of each
    # review, plus the opening if available. 
    md_path = out_dir / "sample_reviews_per_cluster.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Sample Reviews by Opening Class\n\n")
        f.write(f"{CONFIG['n_samples']} randomly drawn reviews per cluster "
                f"(seed={seed}, reproducible).\n\n")
        for cid, grp in samp.groupby("cluster", sort=True):
            f.write(f"\n## Cluster {cid} — {grp['cluster_name'].iloc[0]}\n\n")
            for i, (_, r) in enumerate(grp.iterrows(), 1):
                f.write(f"**{i}.** ({r['star_rating']:.0f} stars, "
                        f"{int(r['helpfulness_votes'])} votes, "
                        f"{int(r['word_count'])} words)\n\n")
                if cols["opening"]:
                    f.write(f"> *Opening:* {str(r[cols['opening']]).strip()}\n\n")
                body = " ".join(str(r[cols["text"]]).split())
                f.write(f"{body[:600]}{'…' if len(body) > 600 else ''}\n\n")

    print(f"  -> {csv_path}")
    print(f"  -> {md_path}")
    return samp


# ---------------------------------------------------------------------------
# 5. Deliverable 2a — total sample stats, then per-cluster stats
# ---------------------------------------------------------------------------

def statistics(df, cols, out_dir: Path) -> pd.DataFrame:
    def row(label, sub, cid="ALL"):
        return {
            "cluster_id": cid,
            "cluster_name": label,
            "n_reviews": len(sub),
            "pct_of_total": round(100 * len(sub) / len(df), 1),
            "mean_word_count": round(sub[cols["words"]].mean(), 1),
            "median_word_count": round(sub[cols["words"]].median(), 1),
            "mean_star_rating": round(sub[cols["rating"]].mean(), 2),
            "median_star_rating": round(sub[cols["rating"]].median(), 1),
            "mean_helpful_votes": round(sub[cols["votes"]].mean(), 2),
            "median_helpful_votes": round(sub[cols["votes"]].median(), 1),
            "sd_helpful_votes": round(sub[cols["votes"]].std(), 2),
            "max_helpful_votes": int(sub[cols["votes"]].max()),
        }

    # Total sample first, exactly as requested, then each cluster.
    rows = [row("TOTAL SAMPLE", df)]
    for cid, grp in df.groupby(cols["cluster"], sort=True):
        rows.append(row(grp[cols["clustnm"]].iloc[0], grp, cid))

    stats = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cluster_statistics.csv"
    stats.to_csv(path, index=False, encoding="utf-8")

    banner("BASIC STATISTICS — TOTAL SAMPLE, THEN CLUSTERS")
    show = stats[["cluster_id", "cluster_name", "n_reviews", "pct_of_total",
                  "mean_word_count", "median_word_count", "mean_star_rating",
                  "mean_helpful_votes", "median_helpful_votes"]]
    print(show.to_string(index=False))
    print(f"\n  -> {path}")
    return stats


# ---------------------------------------------------------------------------
# 6. Deliverable 2b — distribution plots, Stage 1 style
# ---------------------------------------------------------------------------

def plots(df, cols, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_ids = sorted(df[cols["cluster"]].unique())
    names = {c: df.loc[df[cols["cluster"]] == c, cols["clustnm"]].iloc[0] for c in cluster_ids}
    palette = plt.cm.tab10(np.linspace(0, 1, 10))[: len(cluster_ids)]
    wcap, vcap = CONFIG["wordcount_cap"], CONFIG["votes_cap"]

    # --- Figure 1: overlay, one line per cluster, densities so unequal cluster
    #     sizes stay comparable. Mirrors Stage 1 Figure 1's three panels.
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    ratings = sorted(df[cols["rating"]].dropna().unique())
    width = 0.8 / len(cluster_ids)
    for i, c in enumerate(cluster_ids):
        sub = df[df[cols["cluster"]] == c]
        prop = [(sub[cols["rating"]] == r).mean() for r in ratings]
        ax[0].bar(np.arange(len(ratings)) + i * width - 0.4 + width / 2,
                  prop, width, color=palette[i], label=f"{c}: {names[c]}")
    ax[0].set_xticks(range(len(ratings)))
    ax[0].set_xticklabels([f"{int(r)}" for r in ratings])
    ax[0].set_xlabel("Star rating")
    ax[0].set_ylabel("Proportion within cluster")
    ax[0].set_title("Star rating distribution by cluster")

    bins_w = np.linspace(0, wcap, 41)
    for i, c in enumerate(cluster_ids):
        sub = df[df[cols["cluster"]] == c]
        ax[1].hist(sub[cols["words"]].clip(upper=wcap), bins=bins_w, density=True,
                   histtype="step", linewidth=1.6, color=palette[i])
    ax[1].set_xlabel(f"Review word count (capped at {wcap})")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Review length distribution by cluster")

    bins_v = np.linspace(0, vcap, 26)
    for i, c in enumerate(cluster_ids):
        sub = df[df[cols["cluster"]] == c]
        ax[2].hist(sub[cols["votes"]].clip(upper=vcap), bins=bins_v, density=True,
                   histtype="step", linewidth=1.6, color=palette[i])
    ax[2].set_xlabel(f"Helpfulness votes (capped at {vcap})")
    ax[2].set_ylabel("Density")
    ax[2].set_title("Helpfulness vote distribution by cluster")

    for a in ax:
        a.grid(alpha=0.25)
        a.set_axisbelow(True)

    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center",
               ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Figure 1. Distributions by opening class, all clusters overlaid",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    p1 = out_dir / "cluster_distributions_overlay.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: small multiples. Eight overlaid curves are hard to read;
    #     this version is the one to put in the report. Grey = total sample,
    #     so every cluster is visibly compared against the corpus baseline.
    n = len(cluster_ids)
    fig, ax = plt.subplots(n, 3, figsize=(14, 2.1 * n), sharex="col")
    if n == 1:
        ax = ax.reshape(1, 3)

    for i, c in enumerate(cluster_ids):
        sub = df[df[cols["cluster"]] == c]

        base = [(df[cols["rating"]] == r).mean() for r in ratings]
        prop = [(sub[cols["rating"]] == r).mean() for r in ratings]
        ax[i, 0].bar(range(len(ratings)), base, color="0.85", label="Total sample")
        ax[i, 0].bar(range(len(ratings)), prop, color=palette[i], alpha=0.75, label="Cluster")
        ax[i, 0].set_xticks(range(len(ratings)))
        ax[i, 0].set_xticklabels([f"{int(r)}" for r in ratings])
        ax[i, 0].set_ylabel(f"{c}: {names[c]}", fontsize=9, rotation=0,
                            ha="right", va="center", labelpad=10)

        ax[i, 1].hist(df[cols["words"]].clip(upper=wcap), bins=bins_w,
                      density=True, color="0.85")
        ax[i, 1].hist(sub[cols["words"]].clip(upper=wcap), bins=bins_w,
                      density=True, color=palette[i], alpha=0.75)

        ax[i, 2].hist(df[cols["votes"]].clip(upper=vcap), bins=bins_v,
                      density=True, color="0.85")
        ax[i, 2].hist(sub[cols["votes"]].clip(upper=vcap), bins=bins_v,
                      density=True, color=palette[i], alpha=0.75)

        for j in range(3):
            ax[i, j].grid(alpha=0.25)
            ax[i, j].set_axisbelow(True)
            ax[i, j].tick_params(labelsize=8)

    ax[0, 0].set_title("Star rating", fontsize=11)
    ax[0, 1].set_title(f"Word count (capped {wcap})", fontsize=11)
    ax[0, 2].set_title(f"Helpfulness votes (capped {vcap})", fontsize=11)
    ax[0, 0].legend(fontsize=8, loc="upper left")
    ax[-1, 0].set_xlabel("Stars")
    ax[-1, 1].set_xlabel("Words")
    ax[-1, 2].set_xlabel("Votes")
    fig.suptitle("Figure 2. Each cluster (colour) against the total sample (grey)",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    p2 = out_dir / "cluster_distributions_by_class.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  -> {p1}")
    print(f"  -> {p2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4 cluster output report")
    ap.add_argument("--corpus", type=Path, default=CONFIG["corpus_path"])
    ap.add_argument("--taxonomy", type=Path, default=CONFIG["taxonomy_path"])
    ap.add_argument("--inspect", action="store_true",
                    help="print the columns of both files and exit")
    ap.add_argument("--n", type=int, default=CONFIG["n_samples"],
                    help="reviews to sample per cluster (default 10)")
    args = ap.parse_args()

    try:
        if args.inspect:
            inspect(args.corpus, args.taxonomy)
            return 0

        banner("STAGE 4 — CLUSTER OUTPUT REPORT")
        corpus, taxonomy = load(args.corpus, args.taxonomy)

        merged = merge(corpus, taxonomy)
        df, cols = prepare(merged)

        banner("1. SAMPLE REVIEWS PER CLUSTER")
        sample_reviews(df, cols, CONFIG["stats_out"], args.n, CONFIG["seed"])

        statistics(df, cols, CONFIG["stats_out"])

        banner("2. DISTRIBUTION PLOTS")
        plots(df, cols, CONFIG["plots_out"])

        banner("DONE")
        return 0

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        print("Run `python stage4_output.py --inspect` to see the actual column "
              "names and row counts in your two files.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
