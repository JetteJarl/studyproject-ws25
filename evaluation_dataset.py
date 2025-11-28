from typing import List

import pandas as pd

from llm_model import generate_answer
from rag_pipeline import load_rag

# Build a lightweight reference text. If you can’t extract evidence sentences,
# at least encode the label in the ground truth to help 'correctness'.
def _ref_text(row) -> str:
    evidence = f"The claim is {row['label'].lower()} according to Wikipedia evidence."
    return evidence

def load_fever_split(sample_size: int = 200, seed: int = 7) -> pd.DataFrame:
    """
    Load a FEVER split and prepare a DataFrame with columns:
    - user_input: the claim
    - ground_truth: evidence text or label-based reference
    - label: SUPPORTS/REFUTES/NOT ENOUGH INFO
    We filter out NOT ENOUGH INFO for clean supervision.
    """
    # Newer versions of dataset contain a bug that prevents loading datasets remotely:
    # https://github.com/huggingface/datasets/issues/7693
    # Downloaded the data manually from https://fever.ai/dataset/fever.html and saved it locally
    df = pd.read_json("data/shared_task_dev.jsonl", lines=True)
    # Keep only entries with label != NEI
    df = df[df["label"].isin(["SUPPORTS", "REFUTES"])].copy()

    df["user_input"] = df["claim"].astype(str)
    df["ground_truth"] = df.apply(_ref_text, axis=1)

    # Sample for a quick evaluation run
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)

    # Keep only necessary columns
    df = df[["user_input", "ground_truth", "label"]].reset_index(drop=True)
    return df

def run_pipeline_on_querys(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """
    Calls your RAG pipeline for each query.
    Expects load_rag(query) -> (answer: str, contexts: List[str], llm: str, embedder: str).
    """
    answers: List[str] = []
    contexts_list: List[List[str]] = []

    retriever, chain, llm, embedder = load_rag()

    for query in df["user_input"].tolist():
        answer, contexts = generate_answer(query, retriever, chain)
        answers.append(answer)
        # Ensure contexts is a list of strings
        clean_contexts = []
        for c in contexts or []:
            if hasattr(c, "page_content"):
                clean_contexts.append(c.page_content.strip())
            else:
                clean_contexts.append(str(c).strip())
        contexts_list.append(clean_contexts)

    out = df.copy()
    out["answer"] = answers
    out["contexts"] = contexts_list
    return out, llm, embedder