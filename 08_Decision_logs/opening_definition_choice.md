# Opening Definition Choice — Stage 2

## Date: April 2025 (Updated: Session 4)

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

Status: **Confirmed by Prof. Mora (Session 4)**

## Rationale

1. **More syntactic material for parsing.** Ten words captures most opening constructions in their entirety, giving Benepar constituency parsing more structure to work with than 5 words, which often truncates mid-phrase.
2. **Near-uniform length preserves clustering validity.** Standard deviation of 0.24 avoids the clustering contamination problem that would arise with first-sentence openings (std = 11.24), where KMeans would cluster by length rather than by syntactic pattern.
3. **Extends Mora & Izadi (2024) rather than replicating it.** The foundation paper used first 5 words on Yelp reviews. Expanding to 10 words on Amazon reviews represents a methodological contribution — testing whether register signatures persist in a longer opening window — rather than simple replication.
4. **The opening remains a signal of full-text register, not a cause of helpfulness.** Per Mora & Izadi Study 4, swapping only the opening does not change perceived helpfulness. Whether the opening is 5 or 10 words does not alter this associational framing.

## Change History

| Date | Definition | Status | Notes |
|------|-----------|--------|-------|
| Session 2 | First 5 words | Default | Set pending advisor confirmation |
| Session 4 | First 10 words | **Confirmed** | Prof. Mora confirmed after reviewing Stage 3 progress report |

## Pipeline Impact

Switching from 5 to 10 words requires:
- **Stage 1:** Evaluate whether minimum review length filter should increase from >= 5 words to >= 10 words (pending advisor decision — see filtering_decisions.md)
- **Stage 2:** Change primary opening column from `open_5w` to `open_10w`; rerun spaCy and Benepar parsing
- **Stage 3:** Regenerate templates and embeddings from new 10-word parses

Code change is minimal: `df['opening'] = df['open_10w']` instead of `df['opening'] = df['open_5w']`

## PRD Flag

OPENING_DEFINITION: **first 10 words** (confirmed by Prof. Mora, Session 4)
