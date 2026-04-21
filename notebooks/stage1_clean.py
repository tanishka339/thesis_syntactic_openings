import pandas as pd
import gzip
import json
import os 

BASE_DIR = r"C:\Users\tpasumarthi\thesis_syntactic_openings"
DATA_DIR = os.path.join(BASE_DIR, "00_data")
OUT_DIR = os.path.join(BASE_DIR, "01_processed")
BOOKS_FILE = os.path.join(DATA_DIR, "Books_5.json.gz")
ELECTRONICS_FILE = os.path.join(DATA_DIR, "Electronics_5.json.gz")

MIN_REVIEW_WORDS = 5
MIN_HELPFUL_VOTES = 1
SAMPLE_SIZE = 5000
RANDOM_SEED = 42

def load_raw(filepath, domain):
    print(f"Loading {domain} from {filepath} ...")
    records = []
    limit = 60000
    with gzip.open(filepath, "rb") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    df = pd.DataFrame(records)   
    print(f"  Loaded {len(df)} reviews.")
    return df

#filter 1 - length filter
def filter_length(df, domain, log):
    before = len(df)
    df = df[df["reviewText"].notna()]
    df = df[df["reviewText"].str.split().str.len() >= MIN_REVIEW_WORDS]
    after = len(df)
    log.append(f"[{domain}] Length filter: {before - after} removed, {after} remaining.")
    return df

#filter 2 - helpfulness vote filter
def filter_helpfulness(df, domain, log):
    before = len(df)
    df["vote"] = pd.to_numeric(df["vote"], errors="coerce")
    df = df[df["vote"] >= MIN_HELPFUL_VOTES]
    after = len(df)
    log.append(f"[{domain}] Helpfulness filter: {before - after} removed, {after} remaining.")
    return df

#filter 3 - language filter (fast ASCII ratio method)
def filter_language(df, domain, log):
    before = len(df)
    def is_english(text):
        if not isinstance(text, str) or len(text) == 0:
            return False
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / len(text) > 0.90
    df = df[df["reviewText"].apply(is_english)]
    after = len(df)
    log.append(f"[{domain}] Language filter: {before - after} removed, {after} remaining.")
    return df

def sample_corpus(df, domain, log):
    before = len(df)
    if before <= SAMPLE_SIZE:
        log.append(f"[{domain}] Sampling: only {before} reviews available, keeping all.")
        return df
    df = df.groupby("overall", group_keys=False).apply(
        lambda x: x.sample(frac=SAMPLE_SIZE/before, random_state=RANDOM_SEED)
        ).head(SAMPLE_SIZE)
    log.append(f"[{domain}] Sampling: {before} -> {len(df)} reviews sampled.")
    return df
    
def process_domain(filepath, domain):
    log = []
    df = load_raw(filepath, domain)
    df = filter_length(df, domain, log)
    df = filter_helpfulness(df, domain, log)
    df = filter_language(df, domain, log)
    df = sample_corpus(df, domain, log)
    df["domain"] = domain
    return df, log

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    books_df, books_log = process_domain(BOOKS_FILE, "Books")
    elec_df, elec_log = process_domain(ELECTRONICS_FILE, "Electronics")
    corpus = pd.concat([books_df, elec_df], ignore_index=True)
    out_path = os.path.join(OUT_DIR, "corpus.parquet")
    corpus.to_parquet(out_path, index=False)
    print(f"Saved corpus to {out_path} ({len(corpus)} rows).")
    all_logs = books_log + elec_log
    log_path = os.path.join(OUT_DIR, "filtering_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(all_logs))
    print(f"Saved filtering log to {log_path}.")
if __name__ == "__main__":
    main()
    

