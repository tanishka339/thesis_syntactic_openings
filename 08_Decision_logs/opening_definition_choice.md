# Opening Definition Choice — Stage 2

## Date: April 2025

## Definitions Tested

Three candidate opening definitions were evaluated on the 5,000-review development corpus (9,999 total reviews: 4,999 Books + 5,000 Electronics):

### Definition A — First 5 words
- Mean: 5.0 | Std: 0.0 | Min: 5.0 | Max: 5.0
- Perfectly uniform length across all reviews

### Definition B — First sentence (NLTK sent_tokenize)
- Mean: 16.84 | Std: 11.24 | Min: 1 | Max: 187
- Captures complete syntactic units but extreme length variability

### Definition C — First 10 words
- Mean: 9.98 | Std: 0.24 | Min: 5.0 | Max: 10.0
- Near-uniform length, more syntactic material than 5 words

## Decision

Primary definition: **First 10 words (Definition C)**

Status: **Confirmed by Prof. Mora**

## Rationale

1. Alignment with Mora & Izadi (2024), who used first 5 words on Yelp reviews
2. Zero length variance eliminates clustering contamination — KMeans would cluster by length rather than syntactic pattern if lengths vary widely
3. The opening is a signal of full-text register, not a standalone cause of helpfulness (per Study 4) — completeness of the syntactic unit is not required
4. Direct comparability with the foundation paper strengthens the thesis contribution

## Alternative Under Consideration

First 10 words (Definition C) remains viable:
- Near-uniform length (std = 0.24) avoids the clustering contamination problem
- More syntactic material for benepar to produce meaningful constituency trees
- Prof. Mora has expressed openness to expanding beyond 5 words

Switching from 5 to 10 words requires changing one line of code:
`df['opening'] = df['open_10w']` instead of `df['opening'] = df['open_5w']`

## PRD Flag

OPENING_DEFINITION: currently set to first 5 words. Update PRD if advisor confirms change to 10 words.