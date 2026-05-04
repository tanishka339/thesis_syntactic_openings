# Filtering Decisions — Stage 1

## Filter 1: Minimum Review Length
- **Rule:** reviewText must contain >= 5 words
- **Rationale:** Originally set when opening was defined as first 5 words per Mora & Izadi (2024). Reviews shorter than 5 words have no complete opening and cannot be parsed in Stage 2.
- **Books removed:** 6,062 | **Electronics removed:** 8,695

### ⚠️ FLAG: Review Needed After Opening Definition Change (Session 4)
Opening definition has been confirmed as **first 10 words** by Prof. Mora. With the current >= 5 word filter, some reviews in the corpus have only 5–9 words, meaning their 10-word opening is truncated (open_10w min = 5.0, mean = 9.98, std = 0.24). Two options:

| Option | Pros | Cons |
|--------|------|------|
| Keep >= 5 words | Preserves corpus size; truncated openings are very few | Some openings are shorter than 10 words, introducing slight length variation |
| Raise to >= 10 words | All openings are exactly 10 words; uniform length | Loses a small number of reviews; requires Stage 1 rerun |

**Decision needed from Prof. Mora before rerunning Stage 1.**

## Filter 2: Minimum Helpfulness Votes
- **Rule:** vote field must be >= 1
- **Rationale:** Zero-vote reviews provide no helpfulness signal for Stage 6 regression. vote field coerced from string to numeric using pd.to_numeric(errors='coerce'). NaN values fail the >= 1 check and are dropped automatically.
- **Books removed:** 44,252 | **Electronics removed:** 41,064

## Filter 3: Language Detection
- **Rule:** ASCII ratio > 0.90
- **Rationale:** Reviews must be English. ASCII ratio method checks fraction of characters within standard 128-character ASCII range. Faster than langdetect inference — arithmetic vs statistical model. Applied last to minimise rows processed.
- **Books removed:** 0 | **Electronics removed:** 0

## Filter Ordering Rationale
Filters ordered by computational cost: arithmetic (length) before type coercion (helpfulness) before character ratio (language). This minimises processing time on the full corpus.
