# Syntactic Openings in Online Reviews

**A Computational Pipeline for Cross-Domain Register Taxonomy and Helpfulness Analysis**

MSc Data Science Thesis — Tanishka Nath Pasumarthi  
Charlton College of Business, UMass Dartmouth  
Advisor: [Prof. José Domingo Mora](mailto:josedomingo.mora@umassd.edu)

---

## Abstract

Online reviews are consumed in volume and evaluated rapidly, making the linguistic properties of their earliest words consequential. Mora and Izadi (2024) demonstrated that the grammatical and syntactic composition of a review's opening carries diagnostic information about the register of the full text, and that this register co-occurs with perceived helpfulness. Critically, their fourth study established a causal null: substituting only the opening while holding the remaining text constant does not alter perceived helpfulness. The opening is therefore best understood as a signal of register rather than a driver of reader judgment.

This thesis operationalizes, evaluates, and extends that account through a reproducible seven-stage computational pipeline applied to **9,999 Amazon reviews** drawn equally from the **Books** and **Electronics** domains. Review openings — operationally defined as the first ten words — were parsed for dependency and constituency structure using spaCy and benepar with a zero-percent parse failure rate. Parse output was abstracted into canonical syntactic templates, embedded as 384-dimensional sentence vectors, reduced via principal component analysis, and clustered using k-means. An **eight-class taxonomy of opening strategies** was selected based on silhouette score, Davies–Bouldin index, and stability across random initializations (mean adjusted Rand index = 0.99). The taxonomy was validated against manual annotation of 250 openings and tested for association with helpfulness using nested negative binomial regression and for cross-domain generalizability using chi-square and Kruskal–Wallis tests.

Opening class was significantly associated with helpfulness after controlling for review length, star rating, domain, and reviewer activity, and this association varied by domain, yet it explained less than one percent of additional variance. Taxonomy composition was broadly stable across domains (Cramér's V = 0.086). Together these findings corroborate the signal account: a diagnostic marker should be statistically detectable yet weak as a standalone predictor. The thesis contributes an automated, evaluated, and reproducible alternative to semi-manual register analysis.

## Research Questions

| RQ | Question | Pipeline Stage |
|----|----------|----------------|
| **RQ1** | What is the best operational definition of a review opening? | Stage 2 |
| **RQ2** | What syntactic structures appear most commonly in review openings across Books and Electronics? | Stages 2–3 |
| **RQ3** | Can openings be automatically grouped into semantically coherent classes, and how reliable is that classification? | Stages 4–5 |
| **RQ4** | Do opening classes co-occur with helpfulness, and does this pattern differ between domains? | Stages 6–7 |

## Opening Class Taxonomy

| ID | Class | n (%) | Description |
|----|-------|-------|-------------|
| 0 | Product Reference | 1,387 (13.9%) | Demonstrative reference to item — *This/The/A [product]…* |
| 1 | Acquisition Narrative | 1,060 (10.6%) | Past-tense purchasing or obtaining account |
| 2 | Evaluative Positioning | 807 (8.1%) | Complex clauses framing stance — conditionals, contrasts, meta-commentary |
| 3 | Direct Evaluation | 2,142 (21.4%) | Opinion-first with immediate judgment |
| 4 | Specific Identification | 1,012 (10.1%) | Model numbers, dates, proper nouns — information-dense |
| 5 | Affective Stance | 1,169 (11.7%) | First-person emotional framing before product details |
| 6 | Contextual Framing | 1,434 (14.3%) | Background-setting and contextualizing before evaluation |
| 7 | Terse Evaluation | 988 (9.9%) | Fragment-heavy, compressed judgments |

## Pipeline

```
Stage 1 ─ Data Collection & Filtering
       │   Amazon 5-core reviews → langdetect, ASCII filter, min-length, min-vote
       ▼
Stage 2 ─ Opening Extraction
       │   First 10 words per review (confirmed with advisor)
       ▼
Stage 3 ─ Syntactic Parsing & Embedding
       │   spaCy + benepar parse → constituency & coarse-POS templates
       │   all-MiniLM-L6-v2 → 384-dim sentence vectors
       ▼
Stage 4 ─ Clustering & Taxonomy Construction
       │   PCA (50 components, 97.7% variance) → KMeans (k=8)
       │   Evaluated via silhouette, Davies-Bouldin, ARI stability (0.99)
       ▼
Stage 5 ─ Annotation Validation
       │   250-opening stratified sample, blind manual annotation
       │   Cohen's κ = 0.18 — reveals semantic-functional (not purely syntactic) organization
       ▼
Stage 6 ─ Negative Binomial Regression
       │   3 nested models: controls → + class dummies → + class × domain interactions
       │   Best model: Model 3 (AIC = 66,828)
       ▼
Stage 7 ─ Cross-Domain Analysis
           Chi-square (Cramér's V = 0.086), Kruskal-Wallis with BH correction
           Taxonomy composition stable across Books and Electronics
```

## Key Results

- **Opening class is significantly associated with helpfulness** after controlling for review length, star rating, domain, and reviewer activity — but explains < 1% additional variance, consistent with the signal-not-cause framing from Mora & Izadi (2024).
- **Two domain-specific interactions**: Acquisition Narrative and Terse Evaluation behave differently in Books vs. Electronics.
- **Cross-domain stability**: the eight-class taxonomy composition is broadly consistent across domains (Cramér's V = 0.086).
- **Annotation validation** (κ = 0.18) reveals that the embedding-based pipeline captures semantic-functional similarity rather than strict syntactic categories — a finding about what sentence-transformer embeddings encode, not a pipeline failure.
- **Full reproducibility confirmed**: pipeline reproduced byte-identical results on a different machine from intermediate files using fixed seeds throughout.

## Project Structure

```
00_data/              Raw Amazon review data (never modified)
01_processed/         Cleaned corpus (parquet)
02_openings/          Extracted 10-word opening segments
03_parsed/            spaCy + benepar parse outputs
04_templates/         Canonical syntactic labels (constituency + coarse POS)
05_embeddings/        384-dim sentence-transformer vectors
06_clusters/          PCA, KMeans evaluation, silhouette/DB plots
07_taxonomy/          Final 8-class assignments + cross-domain comparison outputs
08_annotation/        Annotation guide, sample, validation metrics
08_Decision_Logs/     Documented decisions at each stage
09_regression/        Negative binomial model summaries + coefficient plots
10_writing/           Thesis chapter drafts
notebooks/            One script per pipeline stage
```

## Setup

```bash
conda activate thesis_env
pip install -r requirements.txt
```

### Core Dependencies

- **Parsing**: spaCy, benepar
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Clustering**: scikit-learn (PCA, KMeans, silhouette, ARI)
- **Regression**: statsmodels (NegativeBinomial)
- **Data**: pandas, numpy, pyarrow

## Technical Parameters

| Parameter | Value | Stage |
|-----------|-------|-------|
| Opening definition | First 10 words | 2 |
| Embedding model | all-MiniLM-L6-v2 (384-dim) | 3 |
| PCA components | 50 (97.7% variance retained) | 4 |
| Cluster count (k) | 8 | 4 |
| KMeans random state | 42 (n_init=10) | 4 |
| Stability (ARI) | 0.9875 across 10 runs | 4 |
| Annotation sample | 250 (125 Books + 125 Electronics) | 5 |
| Regression family | Negative binomial (var/mean = 81.4) | 6 |
| Reference class | Direct Evaluation (n=2,142) | 6 |

## Data

Amazon Reviews 5-core datasets (Books and Electronics) from the [Ni et al. curated collection](https://nijianmo.github.io/amazon/). The 5-core versions require at least 5 reviews per product. Corpus: **4,999 Books + 5,000 Electronics = 9,999 reviews**.

Filtering: English-language only (ASCII ratio > 0.90), minimum 5 words, minimum 1 helpfulness vote, stratified sampling with `random_state=42`.

## Reference

This thesis extends:

> Mora, J. D., & Izadi, A. (2024). Off to a Good Start: Grammar and Syntax in the Opening Predict Review Helpfulness.

## License

This repository contains thesis research code. Please contact the author before reuse.

## Contact

Tanishka Nath Pasumarthi — [tpasumarthi@umassd.edu](mailto:tpasumarthi@umassd.edu)
