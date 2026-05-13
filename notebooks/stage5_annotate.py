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

# ── Part 2: Validation metrics (run AFTER annotation is complete) ──

# Uncomment the lines below after you have completed annotation
# and saved the file as 08_annotation/annotated.csv

# from sklearn.metrics import classification_report, cohen_kappa_score

# annotated = pd.read_csv('08_annotation/annotated.csv', encoding='utf-8')

# # Check all rows are labeled
# assert annotated['manual_label'].notna().all(), "Some rows are missing manual labels!"
# assert len(annotated) == 250, f"Expected 250 rows, got {len(annotated)}"

# gold = annotated['manual_label'].astype(int)
# predicted = annotated['cluster'].astype(int)

# # Per-class precision, recall, F1
# class_names = [
#     '0-Product Reference', '1-Acquisition Narrative',
#     '2-Evaluative Positioning', '3-Direct Evaluation',
#     '4-Specific Identification', '5-Affective Stance',
#     '6-Contextual Framing', '7-Terse Evaluation'
# ]
# report = classification_report(gold, predicted, target_names=class_names)
# print(report)

# # Save classification report
# with open('08_annotation/validation_results.csv', 'w', encoding='utf-8') as f:
#     f.write(classification_report(gold, predicted, target_names=class_names, output_dict=False))

# # Cohen's kappa
# kappa = cohen_kappa_score(gold, predicted)
# print(f"\nCohen's kappa: {kappa:.4f}")

# # Save kappa result
# with open('08_annotation/kappa_result.txt', 'w', encoding='utf-8') as f:
#     f.write(f"Cohen's kappa: {kappa:.4f}\n\n")
#     f.write("Interpretation:\n")
#     if kappa >= 0.81:
#         f.write("Almost perfect agreement (Landis & Koch, 1977)\n")
#     elif kappa >= 0.61:
#         f.write("Substantial agreement (Landis & Koch, 1977)\n")
#     elif kappa >= 0.41:
#         f.write("Moderate agreement (Landis & Koch, 1977)\n")
#     elif kappa >= 0.21:
#         f.write("Fair agreement (Landis & Koch, 1977)\n")
#     else:
#         f.write("Slight or poor agreement (Landis & Koch, 1977)\n")

# print("\nSaved: 08_annotation/validation_results.csv")
# print("Saved: 08_annotation/kappa_result.txt")