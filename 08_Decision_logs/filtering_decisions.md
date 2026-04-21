# Filtering Decisions — Stage 1

## Filter 1: Minimum Review Length
- **Rule:** reviewText must contain >= 5 words
- **Rationale:** Opening is defined as first 5 words per Mora & Izadi (2024). Reviews shorter than 5 words have no complete opening and cannot be parsed in Stage 2.
- **Books removed:** 6,062 | **Electronics removed:** 8,695

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