import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_parquet(r"C:\Users\tpasumarthi\thesis_syntactic_openings\02_openings\parsed_openings.parquet")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(df["parse_tree"].iloc[0])

def extract_template(tree_str):
   template = re.sub(r' [^()]+(?=\))', '', tree_str)
   return template

COARSE_MAP = {
   "NOUN": "N", "PROPN": "N",
   "VERB": "V", "AUX": "V",
   "ADJ": "ADJ", "ADV": "ADV",
   "PRON": "PRN", "DET": "DET", "ADP": "PREP",
   "CCONJ": "CONJ", "SCONJ": "CONJ",
   "PART": "PART", "PUNCT": "PUNCT",
   "NUM": "NUM", "INTJ": "INTJ",
   "X": "X", "SYM": "X"
   }

def coarsen(pos_array):
   return " ".join(COARSE_MAP.get(tag, "X") for tag in pos_array)

print(extract_template(df["parse_tree"].iloc[0]))
df["template"] = df["parse_tree"].apply(extract_template)

df["coarse_template"] = df["pos_tags"].apply(coarsen)
n_coarse = df["coarse_template"].nunique()
print(f"\nCoarse templates: {n_coarse} unique out of {len(df)} openings")
print(f"Constituency templates: {df['template'].nunique()} unique out of {len(df)} openings")
print(f"\nTop 10 coarse templates:")
print(df["coarse_template"].value_counts().head(10))
print(df["template"].value_counts().head(10))

print(f"\nSample POS sequences:")
print(df["pos_tags"].head(5))

print(type(df["pos_tags"].iloc[0]))
print(repr(df["pos_tags"].iloc[0]))

pos_sequences = df["pos_tags"].apply(lambda x: " ".join(x))
print(pos_sequences.iloc[0])

print("\nLoading sentence-transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding POS sequences...")
embeddings = model.encode(pos_sequences.tolist(), show_progress_bar=True, batch_size=64)
print(f"Embeddings shape: {embeddings.shape}")

np.save(r"C:\Users\tpasumarthi\thesis_syntactic_openings\05_embeddings\embeddings.npy", embeddings)
print(f"Saved embeddings to 05_embeddings/embeddings.npy")

df.to_parquet(r"C:\Users\tpasumarthi\thesis_syntactic_openings\04_templates\openings_with_templates.parquet", index=False)
print(f"Saved dataframe with templates to 04_templates/openings_with_templates.parquet")
print(f"Final dataframe: {len(df)} rows, {len(df.columns)} columns")

df[["reviewerID"]].to_csv(r"C:\Users\tpasumarthi\thesis_syntactic_openings\05_embeddings\index.csv", index=False)
print("Saved index.csv")

top_templates = df["coarse_template"].value_counts().head(50).reset_index()
top_templates.columns = ["coarse_template", "count"]
top_templates.to_csv(r"C:\Users\tpasumarthi\thesis_syntactic_openings\04_templates\top_templates.csv", index=False)
print("Saved top_templates.csv")

from sklearn.metrics.pairwise import cosine_similarity
print("\n=== Nearest Neighbour Spot-Check ===")
sample_indices = [0, 500, 2000, 5000, 8000]
results = []
for idx in sample_indices:
    sims = cosine_similarity([embeddings[idx]], embeddings)[0]
    sims[idx] = -1
    top3 = sims.argsort()[-3:][::-1]
    results.append((idx, top3))
    print(f"\nOpening {idx}: {df['coarse_template'].iloc[idx]}")
    print(f"  POS: {df['pos_tags'].iloc[idx]}")
    for rank, nb in enumerate(top3, 1):
        print(f"  Neighbour {rank} (sim={sims[nb]:.3f}): {df['coarse_template'].iloc[nb]}")
        print(f"    POS: {df['pos_tags'].iloc[nb]}")

with open(r"C:\Users\tpasumarthi\thesis_syntactic_openings\05_embeddings\nearest_neighbour_check.txt", "w") as f:
    for idx, top3 in results:
        f.write(f"\nOpening {idx}: {df['coarse_template'].iloc[idx]}\n")
        f.write(f"  POS: {df['pos_tags'].iloc[idx]}\n")
        sims = cosine_similarity([embeddings[idx]], embeddings)[0]
        sims[idx] = -1
        for rank, nb in enumerate(top3, 1):
            f.write(f"  Neighbour {rank} (sim={sims[nb]:.3f}): {df['coarse_template'].iloc[nb]}\n")
            f.write(f"    POS: {df['pos_tags'].iloc[nb]}\n")
print("Saved nearest_neighbour_check.txt")       