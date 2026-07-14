# Disagreement Analysis — Pipeline vs Human Annotation

## Overview

This document analyzes disagreements between the automated pipeline's cluster assignments and human manual annotation of 250 syntactic openings (125 Books, 125 Electronics). The pipeline achieved an overall accuracy of 28% and Cohen's kappa of 0.1812 (slight agreement, Landis & Koch 1977).

## Key Finding

The low agreement reveals a fundamental difference between what the pipeline captures and what the codebook measures. The pipeline clusters were formed by sentence-transformer embeddings (all-MiniLM-L6-v2), which encode semantic and distributional similarity. The codebook definitions target syntactic structure: verb types, clause complexity, demonstrative reference, sentence completeness. These two organizing principles produce substantially different groupings of the same openings.

## Per-Class Analysis

### Terse Evaluation — Best Agreement (F1 = 0.46)

Terse Evaluation showed the highest agreement. Of 22 human-labeled Terse openings, 13 matched the pipeline (59% recall). This class is defined by fragments, compressed syntax, and non-standard sentence structure — features that are distinctive both syntactically and semantically. Sentence-transformers can detect the distributional signature of compressed language.

### Direct Evaluation — Moderate Agreement (F1 = 0.32)

Direct Evaluation showed a reasonable diagonal (13/32 = 41% recall). However, the pipeline also assigned Direct Evaluation labels to openings the human coded as Contextual Framing (7), Evaluative Positioning (8), and Product Reference (7). The pipeline's Direct Evaluation cluster appears to capture evaluative language broadly, while the codebook restricts it to opinion-first openings with full sentence structure.

### Acquisition Narrative — Moderate Agreement (F1 = 0.36)

Acquisition Narrative had 13/38 agreement (34% recall). The main confusion was with Specific Identification (9 openings) — reviews that mention product names during acquisition stories get pulled toward the information-dense cluster by the embeddings. The pipeline prioritizes the presence of product-specific terms over the acquisition verb structure.

### Contextual Framing — Worst Performing (F1 = 0.31, Largest Class)

Contextual Framing was the most problematic class. The human labeled 63 openings as Contextual Framing, but the pipeline scattered them across all 8 clusters. Only 15 stayed in Contextual Framing (24% recall). The pipeline assigned 14 to Affective Stance and 11 to Acquisition Narrative. This suggests that "providing factual background" is a syntactic judgment that does not correspond to a single semantic cluster — background-setting openings vary widely in their vocabulary and meaning.

### Evaluative Positioning — High Precision, Low Recall (F1 = 0.25)

The human labeled 48 openings as Evaluative Positioning, but only 8 matched the pipeline (17% recall). The precision was higher (0.53), meaning when the pipeline did assign Evaluative Positioning, it often agreed with the human. The pipeline appears to capture a narrow subset of what the codebook defines — perhaps only the most syntactically extreme cases (explicit concessions with "despite" or "although") while missing subtler hedges and conditionals.

### Affective Stance — Lowest F1 (0.12)

Only 3 of 17 human-labeled Affective Stance openings matched the pipeline. The "I + emotion verb" pattern that defines this class syntactically does not form a coherent semantic cluster — "I love this product" and "I was disappointed" share the same syntactic structure but have opposite semantic content, so embeddings pull them into different clusters.

### Product Reference — Low Agreement (F1 = 0.14)

Only 3 of 18 human-labeled Product Reference openings matched the pipeline. The demonstrative + product noun pattern is syntactically uniform but semantically diverse — "This book covers" and "This cable connects" have the same structure but different content.

### Specific Identification — Low Agreement (F1 = 0.16)

Only 3 of 12 human-labeled Specific Identification openings matched the pipeline. Information-dense openings with model numbers and dates get distributed across pipeline clusters based on what comes after the identifying information.

## Systematic Patterns in Disagreement

Three systematic patterns emerge:

**Pattern 1 — Syntactic uniformity, semantic diversity.** Classes defined by a single syntactic feature (Product Reference: demonstrative + noun; Affective Stance: I + emotion verb) contain semantically diverse openings. Embeddings distribute them across multiple clusters based on meaning.

**Pattern 2 — Semantic similarity, syntactic diversity.** The pipeline's clusters group openings that talk about similar things in similar ways, even when their syntactic structures differ. An acquisition narrative about buying headphones and a contextual framing about using headphones may land in the same embedding cluster because they share vocabulary, even though they open differently.

**Pattern 3 — Boundary asymmetry.** The codebook boundary rules assume clear syntactic distinctions (e.g., "if the emotion verb leads, code as Affective Stance"). But embeddings do not encode word order as a primary feature — they encode the overall semantic content of the 10-word window. This means boundary cases that a human resolves by word order are resolved differently by the pipeline.

## Implications

This result does not invalidate the pipeline. It reveals that sentence-transformer embeddings organize review openings along a semantic-functional axis rather than a purely syntactic one. The 8-class taxonomy names assigned in Stage 4 were based on qualitative review of 25 samples per cluster, which gave the impression of syntactic coherence. Systematic annotation shows that the coherence is more semantic than syntactic.

This is consistent with register theory. Register is realized through multiple linguistic features simultaneously — lexical choice, syntactic structure, pragmatic function. Sentence-transformer embeddings may capture this holistic register signal, while the codebook isolates one dimension (syntax). The pipeline's clusters may therefore be better described as functional opening strategies rather than formal syntactic types.

The regression analysis in Stage 6 tests whether the pipeline's clusters — whatever they capture — co-occur with helpfulness patterns. That test remains valid regardless of how the clusters map to human syntactic intuitions.

---

*Stage 5 — Manual Annotation Validation*
*Tanishka Nath Pasumarthi | UMass Dartmouth | June 2026*
