# Cross-Domain Findings (Stage 7)

**Session 8 · Week 3, Day 13 · Books (4,999) vs Electronics (5,000)**

Purpose: test whether the 8-class opening taxonomy and its association with
helpfulness generalise across domains. Framed as **confirmation of Stage 6's
Model 3 by an independent method, not a new model.**

Data: `07_taxonomy/final_classes.parquet` (helpfulness re-derived directly from
its `vote` column; variance/mean ≈ 81.4 reproduces Stage 6 exactly, confirming
`vote` == Stage 6 `helpfulness_count`).

---

## 1. Composition — chi-square + Cramér's V

- Contingency table: 8 classes × 2 domains. Smallest expected cell = **403.5**
  (≫ 5), so chi-square assumptions comfortably met.
- **χ²(7) = 74.44, p = 1.87e-13, Cramér's V = 0.086.**
- Reading: reliable but **negligible** — at n ≈ 10k, p is near-guaranteed;
  the *effect size* is what matters, and V < 0.10 means the class distribution
  is broadly stable across domains.
- Tilts (observed vs expected): Books lean to **Specific Identification**
  (596 vs 506) and **Contextual Framing** (790 vs 717); Electronics lean to
  **Terse Evaluation**, **Acquisition Narrative**, **Affective Stance**.
- Direct Evaluation near-identical across domains (1,072 vs 1,070).

## 2. Association — Kruskal-Wallis per class

- Per class: KW on `vote` by domain; BH-FDR across the 8 tests; epsilon-squared.
- **Epsilon-squared ≤ 0.005 for every class** — trivial throughout.
- Only **Direct Evaluation survives FDR** (p_fdr = .009). It is the regression's
  **reference category** (no interaction term), so this is the **domain main
  effect** (Electronics accrues more votes overall), *not* a class-specific
  pattern. 7 of 8 classes read higher in Electronics — the main effect bleeding
  through the marginal tests.

## 3. FR12 — follow-up on the two Stage 6 interaction classes

Both align **directionally** with Model 3 (mean *and* median agree, so not a
skew artifact):

| Class | Books (mean / med) | Electronics (mean / med) | Higher | Model 3 coef |
|---|---|---|---|---|
| Acquisition Narrative | 9.25 / 4.0 | 14.86 / 5.0 | Electronics | **+0.18** |
| Terse Evaluation | 13.13 / 4.0 | 10.56 / 3.5 | Books | **−0.35** |

Terse Evaluation is the *only* class tilting toward Books — matching the
largest-magnitude interaction from Stage 6. Neither survives FDR: **expected**,
and correct. These marginal tests confirm *direction*, not significance.

---

## Key decisions & reasoning

- **Descriptive, not a re-fit.** Model 3 already tested the interaction
  inferentially with covariates. Stage 7 confirms with a different method.
  Convergence across an *adjusted* (regression) and a *marginal* (KW) view is
  triangulation, not repetition. Requiring the KW tests to clear a corrected
  threshold would be re-testing the same hypothesis.
- **Marginal vs adjusted explains the "mismatches."** Raw gaps and coefficients
  need not rank-order the same way (Acquisition Narrative has the larger raw gap
  but the smaller coefficient) because the coefficient strips the domain main
  effect out and the KW test does not.
- **Medians, not means, in the helpfulness plot.** `vote` is heavily right-skewed
  (SD 30.6, max 643); means track outliers, medians track the typical review.
- **Report effect sizes with every test** (Cramér's V, epsilon-squared) — same
  discipline as Stage 6's pseudo-R².

## Conclusion

Taxonomy composition and its (small) helpfulness association **both generalise**
across Books and Electronics, with modest domain-specific tilts. Uniformly
trivial effect sizes are consistent with the opening as a **diagnostic signal of
register, not a driver of helpfulness** (Mora & Izadi 2024, Study 4). All domain
differences are **co-occurrence patterns** — never causal.

## Flags for Prof. Mora

- **`vote` floored at 2** (min = 2, Q1 = 2; no 0/1 values). Sharpens the pending
  zero-truncated vs standard NB question: Model 3's standard NB puts mass on
  outcomes (0, 1) that cannot occur. Logged; does not affect Stage 7's tests.
- **Min review length filter** (≥5 vs ≥10 words) — still pending.

## Outputs (in `07_taxonomy/domain_comparison/`)

`class_frequency_by_domain.csv` · `kruskal_by_class.csv` ·
`class_frequency_grouped_bar.png` · `helpfulness_by_class_and_domain.png`

→ Feeds PRD v10: Table 9, chi-square + Cramér's V, per-class KW (FDR + ε²), FR12.
