# Decision Log — Regression Model Choice (Stage 6)

**Date:** Stage 6 (Week 3, Session 7)
**Research question:** RQ4 — Do opening classes co-occur with helpfulness patterns, and does the association hold across product domains?

## Dependent variable

`helpfulness_count` (Amazon helpful-vote count per review), an unbounded non-negative count. Corpus filtered to votes ≥ 1 (no zero counts present).

## Model family

- OLS rejected: outcome is a skewed count; OLS can yield negative fitted values and mis-estimates uncertainty.
- Poisson rejected: assumes variance = mean. **Empirical variance-to-mean ratio = 81.4**, indicating severe over-dispersion.
- **Negative binomial adopted.** Dispersion parameter α estimated ≈ 1.02 and significantly > 0 in every model, confirming over-dispersion within the model. (As α → 0 the NB reduces to Poisson.)

## Model specification (three nested models)

- **Model 1 (controls):** `log_wc` (log word count), `overall` + `star_sq` (linear + quadratic star rating), `C(domain)`, `log_exp` (log reviewer activity).
- **Model 2:** Model 1 + `C(cluster_label)` (8 opening classes).
- **Model 3:** Model 2 + `C(cluster_label):C(domain)` interactions.

## Key decisions

- **Reference category:** Direct Evaluation (largest class, n = 2142), so all class coefficients are read against a well-populated baseline.
- **Star rating functional form:** linear + quadratic terms included to allow a non-linear (U-shaped) association; both terms significant, confirming the choice.
- **Reviewer-activity control (`log_exp`):** entered as an ordinary covariate, not a Poisson offset. It is a proxy for reviewer prolificness, not a true reader-exposure measure. Limitation noted; ideal exposure control (review visibility/age) not available in the data.
- **Model comparison:** AIC (primary) plus likelihood-ratio tests for nested comparisons. Both favour Model 3 (opening class adds explanatory value; the association differs by domain).
- **Convergence:** default optimiser (maxiter = 35) failed to converge for Models 2–3; refit with `maxiter=1000`, all three models converged.
- **Interpretation policy:** All coefficients reported as incidence rate ratios and interpreted associationally only ("co-occurs with"), never causally. Consistent with Mora & Izadi (2024), Study 4: the opening is a diagnostic signal of register, not a cause of helpfulness.

## Pending / flag for Prof. Mora

- **Zero-truncation:** because votes ≥ 1, the count distribution has no zeros; a zero-truncated NB would be the textbook-exact specification. Standard NB used here as the common, defensible choice — raise for confirmation.

## Outputs (`09_regression/`)

`model1_summary.csv`, `model2_summary.csv`, `model3_summary.csv`, `aic_comparison.csv`, `coefficient_table.csv`, `coefficient_plot.png`.
