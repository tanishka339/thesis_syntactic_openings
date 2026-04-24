import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import spacy

df = pd.read_parquet(r'C:\Users\tpasumarthi\thesis_syntactic_openings\01_processed\corpus.parquet')
print(df.shape)

def first_k_words(text, k):
    words = text.split() #split text into a list of words
    return ' '.join(words[ :k]) #take first k words, join back inot string

def first_sentence(text): 
    sentences = sent_tokenize(text) # split text into list of sentences
    return sentences[0] #return just the first one

df['open_5w'] = df['reviewText'].apply(lambda x: first_k_words(x, 5))
df['open_sent'] = df['reviewText'].apply(lambda x: first_sentence(x))
df['open_10w'] = df['reviewText'].apply(lambda x: first_k_words(x, 10))
print(df[['reviewText', 'open_5w', 'open_sent', 'open_10w']].head(3))

for i in range(3):
    print(f"\n--- Review {i} ---")
    print(f"FULL:   {df['reviewText'].iloc[i][:80]}...")
    print(f"5-WORD: {df['open_5w'].iloc[i]}")
    print(f"SENT:   {df['open_sent'].iloc[i]}")
    print(f"10-WORD: {df['open_10w'].iloc[i]}")

df['len_5w'] = df['open_5w'].apply(lambda x: len(x.split()))
df['len_sent'] = df['open_sent'].apply(lambda x: len(x.split()))
df['len_10w'] = df['open_10w'].apply(lambda x: len(x.split()))

print("\n=== Length Statistics (word count) ===")
print("\nFirst 5 words:")
print(df['len_5w'].describe())
print("\nFirst sentence:")
print(df['len_sent'].describe())
print("\nFirst 10 words:")
print(df['len_10w'].describe())

df['opening'] = df['open_5w']
print(f"\nPrimary opening set to: first 5 words")
print(f"Example: {df['opening'].iloc[0]}")

print("\nLoading spaCy model...")
nlp = spacy.load('en_core_web_trf')
print("spaCy model loaded.")

def parse_opening(doc):
    pos_tags = [token.pos_ for token in doc]
    dep_labels = [token.dep_ for token in doc]
    head_indices = [token.head.i for token in doc]
    return pos_tags, dep_labels, head_indices

print("\nParsing openings with spaCy...")
results = []
for i, doc in enumerate(nlp.pipe(df['opening'].tolist(), batch_size=64)):
    results.append(parse_opening(doc))
    if (i + 1) % 2000 == 0:
        print(f"  Parsed {i + 1} / {len(df)} openings")

df['pos_tags'] = [r[0] for r in results]
df['dep_labels'] = [r[1] for r in results]
df['head_indices'] = [r[2] for r in results]
print(f"Parsing complete. {len(df)} openings parsed.")

for i in range(3):
    print(f"\n--- Opening {i} ---")
    print(f"TEXT:  {df['opening'].iloc[i]}")
    print(f"POS:  {df['pos_tags'].iloc[i]}")
    print(f"DEP:  {df['dep_labels'].iloc[i]}")
    print(f"HEAD: {df['head_indices'].iloc[i]}")

import benepar
benepar.download('benepar_en3')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
print("Benepar loaded.")

print("\nParsing constituency trees with benepar...")
trees = []
for i, doc in enumerate(nlp.pipe(df['opening'].tolist(), batch_size=64)):
    try:
        tree = list(doc.sents)[0]._.parse_string
        trees.append(tree)
    except:
        trees.append(None)
    if (i + 1) % 2000 == 0:
        print(f"  Parsed {i + 1} / {len(df)} trees")

df['parse_tree'] = trees
print(f"Constituency parsing complete.")
print(f"Null trees: {df['parse_tree'].isna().sum()}")

for i in range(3):
    print(f"\n--- Opening {i} ---")
    print(f"TEXT: {df['opening'].iloc[i]}")
    print(f"TREE: {df['parse_tree'].iloc[i]}")

output_cols = ['reviewerID', 'asin', 'reviewText', 'overall', 'vote',
               'domain', 'opening',
               'open_5w', 'open_sent', 'open_10w',
               'pos_tags', 'dep_labels', 'head_indices', 'parse_tree']

df_out = df[output_cols]
output_path = r'C:\Users\tpasumarthi\thesis_syntactic_openings\02_openings\parsed_openings.parquet'
df_out.to_parquet(output_path)
print(f"\nSaved to {output_path}")
print(f"Shape: {df_out.shape}")
print(f"Columns: {list(df_out.columns)}")
print(list(df.columns))