# Syntactic Openings in Online Reviews
MSc Data Science Thesis — Tanishka Nath Pasumarthi
UMass Dartmouth | Advisor: Prof. José Mora

## Setup
conda activate thesis_env
pip install -r requirements.txt

## Project Structure
00_data/        — raw data only, never modified
01_processed/   — cleaned corpus
02_openings/    — extracted opening segments
03_parsed/      — spaCy + benepar parse outputs
04_templates/   — canonical syntactic labels
05_embeddings/  — sentence-transformer vectors
06_clusters/    — clustering outputs
07_taxonomy/    — final opening class assignments
08_annotation/  — manual annotation files
09_regression/  — statistical model outputs
10_writing/     — thesis chapter drafts
notebooks/      — one notebook per pipeline stage 
