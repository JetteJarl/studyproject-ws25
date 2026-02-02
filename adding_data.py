import sys
import argparse
from pathlib import Path

import pandas as pd

from embed_model import get_embeddings_model
from vector_database import load_vectorstore, add_url, list_current_database
from urllib.parse import urlparse, urlunparse



def read_urls_from_csv(path: str):
    df = pd.read_csv(path, delimiter=',')
    # case-insensitive search for common URL column names
    candidates = ("url", "urls", "link")
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return df[cols_lower[cand]].astype(str).dropna().tolist()
    # fallback: use first column
    return df.iloc[:, 0].astype(str).dropna().tolist()


def normalize_url(u: str) -> str:
    p = urlparse(u.strip())
    scheme = p.scheme.lower() or "http"
    netloc = p.netloc.lower()
    path = p.path.rstrip("/")  # remove trailing slash for comparison
    return urlunparse((scheme, netloc, path, "", "", ""))


def url_in_df_normalized(url: str, df: pd.DataFrame, col: str = "url") -> bool:
    target = normalize_url(url)
    # apply normalization to column (vectorized-ish via .apply)
    return df[col].dropna().astype(str).apply(normalize_url).eq(target).any()


def main():
    parser = argparse.ArgumentParser(description="Ingest CSV of URLs into Chroma vectorstore")
    parser.add_argument("csv", help="Path to CSV file with URLs (column 'url' or first column)")
    parser.add_argument("--emb-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--persist-dir", default="chroma_db")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("CSV file not found:", csv_path)
        sys.exit(1)

    urls = read_urls_from_csv(str(csv_path))
    if not urls:
        print("No URLs found in CSV; The column containing urls should be called 'urls'")
        return

    embeddings = get_embeddings_model(args.emb_model)[0]
    vectorstore = load_vectorstore(embeddings, args.persist_dir)

    # get current data in vectorstore
    current_data = list_current_database(vectorstore)
    df = pd.DataFrame(current_data)  # ensures 'source' column exists

    # ingest URLs one by one
    for url in urls:
        try:
            # check if entry already exists load if not
            if url_in_df_normalized(url, df, col="source"):
                print("Already ingested:", url, " --- skipping ---")
            else:
                vectorstore = add_url(vectorstore, url, embeddings_model=embeddings, persist_directory=args.persist_dir)

            print("Ingested:", url)
        except Exception as e:
            print("Failed to ingest", url, ":", e)

if __name__ == "__main__":
    main()