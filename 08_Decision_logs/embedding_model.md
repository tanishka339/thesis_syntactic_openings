# Decision Log: Embedding Model Choice

**Date:** Session 4 | Stage 3 — Template Abstraction + Semantic Embedding  
**Decision:** Use `all-MiniLM-L6-v2` sentence-transformer to embed POS tag sequences  

---

## Context

Stage 3 requires converting syntactic representations of review openings into dense numerical vectors suitable for unsupervised clustering (KMeans) in Stage 4. The input is a sequence of Universal POS tags extracted by spaCy in Stage 2. The output is a fixed-length vector per opening.

## Decision

**Model:** `all-MiniLM-L6-v2` from the `sentence-transformers` library  
**Embedding dimension:** 384  
**Input:** Space-joined POS tag sequences (e.g., `"PRON VERB DET NOUN"`)  
**Output shape:** (9999, 384)  

## Alternatives Considered

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| all-MiniLM-L6-v2 (384-dim) | Fast, lightweight (~80MB), strong performance on semantic similarity benchmarks, standard in NLP research | Trained on natural language, not POS sequences specifically | **Selected** |
| all-mpnet-base-v2 (768-dim) | Higher benchmark scores, richer representation | Slower, larger model, higher dimensionality may not benefit POS-level input | Rejected — diminishing returns for non-lexical input |
| One-hot / count-based encoding | Simple, fully transparent, no model dependency | Loses sequential ordering; high-dimensional and sparse | Rejected — cannot capture sequential patterns |
| TF-IDF on POS n-grams | Captures local POS patterns, interpretable | Sparse, high-dimensional, no pre-trained semantic knowledge | Rejected — less suitable for KMeans |

## Rationale

1. **POS sequences are short and structurally repetitive.** A lightweight model is sufficient; the additional capacity of larger models (e.g., mpnet-base) would not meaningfully improve separation of POS patterns.
2. **Sequential encoding matters.** `PRON VERB DET NOUN` and `DET NOUN VERB PRON` should have different embeddings. Sentence-transformers preserve token order; bag-of-words approaches do not.
3. **Reproducibility and accessibility.** `all-MiniLM-L6-v2` is the most widely used sentence-transformer in the research community, making results easier to replicate and evaluate.

## Validation

Nearest-neighbour spot-check on 5 sample openings confirmed that structurally similar POS sequences cluster together in embedding space (cosine similarity 0.95–0.99 among nearest neighbours). Full results saved in `05_embeddings/nearest_neighbour_check.txt`.

## What We Embed vs. What We Read

- **Embedded:** POS tag sequences (flat, from `pos_tags` column) — used as input to KMeans clustering.
- **Not embedded:** Constituency tree templates (from `template` column) — used for human interpretation and labeling of clusters after clustering is complete.

These serve parallel roles: embeddings for the algorithm, templates for the analyst.

## PRD Flags

- EMBEDDING_MODEL set to `all-MiniLM-L6-v2` (384-dim).
- BATCH_SIZE set to 64 (PRD specified 256; reduced for stability on available hardware — no effect on output).
- TEMPLATE_APPROACH: two parallel representations created — full-depth constituency templates (9,572 unique) and coarse POS templates (9,798 unique). PRD originally specified coarse POS only; constituency templates added for richer structural detail.
