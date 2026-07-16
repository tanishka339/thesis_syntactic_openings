import pandas as pd

df = pd.read_parquet(r"D:\thesis_syntactic_openings\02_openings\parsed_openings.parquet")
print("shape:", df.shape)
print("columns:", df.columns.tolist())