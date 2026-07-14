import pandas as pd
import os

df = pd.read_parquet('07_taxonomy/final_classes.parquet')
os.makedirs('08_annotation', exist_ok=True)

sample = df.groupby('domain').sample(n=125, random_state=42)

annotation_df = sample[['reviewerID', 'asin', 'domain', 'open_10w', 'cluster', 'cluster_name']].copy()
annotation_df['manual_label'] = ''
annotation_df['confidence'] = ''
annotation_df = annotation_df.reset_index(drop=True)

annotation_df.to_csv('08_annotation/to_annotate.csv', index=False, encoding='utf-8')

print(f"Total sample: {len(annotation_df)}")
print(f"\nDomain breakdown:")
print(annotation_df['domain'].value_counts())
print(f"\nCluster distribution in sample:")
print(annotation_df['cluster_name'].value_counts().sort_index())
print(f"\nSaved to: 08_annotation/to_annotate.csv")