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
print(extract_template(df["parse_tree"].iloc[0]))
df["template"] = df["parse_tree"].apply(extract_template)
n_unique = df["template"].nunique()
print(f"Unique templates: {n_unique} out of {len(df)} openings")
print(f"Top 10 most common templates:")
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