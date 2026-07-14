import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.metrics import confusion_matrix
import numpy as np

annotated = pd.read_csv('08_annotation/annotated.csv', encoding='utf-8')

assert annotated['manual_label'].notna().all(), "Some rows are missing manual labels!"
assert len(annotated) == 250, f"Expected 250 rows, got {len(annotated)}"

gold = annotated['manual_label']
predicted = annotated['cluster_name']

class_names = sorted(gold.unique())

report_text = classification_report(gold, predicted, target_names=class_names, zero_division=0)
print(report_text)

report_dict = classification_report(gold, predicted, target_names=class_names, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv('08_annotation/validation_results.csv', encoding='utf-8')

kappa = cohen_kappa_score(gold, predicted)
print(f"\nCohen's kappa: {kappa:.4f}")

with open('08_annotation/kappa_result.txt', 'w', encoding='utf-8') as f:
    f.write(f"Cohen's kappa: {kappa:.4f}\n\n")
    f.write("Interpretation (Landis & Koch, 1977):\n")
    if kappa >= 0.81:
        f.write("Almost perfect agreement\n")
    elif kappa >= 0.61:
        f.write("Substantial agreement\n")
    elif kappa >= 0.41:
        f.write("Moderate agreement\n")
    elif kappa >= 0.21:
        f.write("Fair agreement\n")
    else:
        f.write("Slight or poor agreement\n")

print("\nSaved: 08_annotation/validation_results.csv")
print("Saved: 08_annotation/kappa_result.txt")

class_names = sorted(gold.unique())
cm = confusion_matrix(gold, predicted, labels=class_names)

print("\nConfusion Matrix (rows = human, columns = pipeline):")
print(f"{'':>25}", end='')
for name in class_names:
    print(f"{name[:8]:>10}", end='')
print()
for i, name in enumerate(class_names):
    print(f"{name:>25}", end='')
    for j in range(len(class_names)):
        print(f"{cm[i][j]:>10}", end='')
    print()