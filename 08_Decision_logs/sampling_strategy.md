# Sampling Strategy — Stage 1

## Approach: Tiered Corpus Strategy

### Development Corpus (current)
- **Sample size:** 5,000 per domain (10,000 total)
- **Load limit:** 60,000 rows per domain from raw file
- **Purpose:** Build and validate full pipeline end to end

### Final Corpus (Stage 6 onwards)
- **Sample size:** 25,000 per domain (50,000 total)
- **Purpose:** Final thesis results and regression analysis

## Stratified Sampling
- **Stratification variable:** overall (star rating 1-5)
- **Method:** groupby('overall').apply(lambda x: x.sample(frac=SAMPLE_SIZE/before))
- **Rationale:** Amazon reviews are skewed toward 5-star ratings. Simple random sampling would produce an unrepresentative corpus. Stratification ensures proportional representation across all rating levels.
- **Random seed:** 42 (fixed for reproducibility)

## Why Not Simple Random Sampling
Without stratification, 5-star reviews would dominate the corpus. This could bias syntactic analysis in Stage 2-4 if positive reviews use systematically different opening structures than negative reviews.

## Final Corpus Statistics
| Domain | N Reviews | Mean Words | Median Words | Mean Rating | Mean Votes |
|--------|-----------|------------|--------------|-------------|------------|
| Books | 4,999 | 215.72 | 148.0 | 3.8 | 10.40 |
| Electronics | 5,000 | 155.70 | 108.0 | 4.0 | 12.65 |
| **Total** | **9,999** | | | | |

## PRD Change Note
Original PRD specified 50,000 per domain. Revised to 5,000 (development) and 25,000 (final) based on computational feasibility and pipeline validation requirements. Decision approved by advisor Prof. Jose Mora.