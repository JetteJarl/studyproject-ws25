from typing import List

import pandas as pd

from llm_model import generate_answer
from rag_pipeline import load_rag

# Build a lightweight reference text. If you can’t extract evidence sentences,
# at least encode the label in the ground truth to help 'correctness'.
def _ref_text(row) -> str:
    """
    Build a lightweight reference/evidence string from a FEVER row.

    Parameters:
        row: A pandas Series containing at least a "label" field.

    Returns:
        A short textual reference encoding the FEVER label, suitable as
        ground truth guidance for correctness-style metrics.
    """
    evidence = f"The claim is {row['label'].lower()} according to Wikipedia evidence."
    return evidence

def load_fever_split(sample_size: int = 200, seed: int = 7) -> pd.DataFrame:
    """
    Load the local FEVER dev split and prepare inputs for evaluation.

    Parameters:
        sample_size: Maximum number of rows to keep (None or 0 to keep all).
        seed: Random seed used when sampling rows.

    Returns:
        DataFrame with columns:
            - user_input: user claim text.
            - ground_truth: lightweight evidence text or label-based reference.
            - label: FEVER label in {"SUPPORTS", "REFUTES"}.
    """
    # Newer versions of dataset contain a bug that prevents loading datasets remotely:
    # https://github.com/huggingface/datasets/issues/7693
    # Downloaded the data manually from https://fever.ai/dataset/fever.html and saved it locally
    df = pd.read_json("data/shared_task_dev.jsonl", lines=True)
    # Keep only entries with label != NEI
    df = df[df["label"].isin(["SUPPORTS", "REFUTES"])].copy()
    # Rename column because Ragas expects "user_input"
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
    Execute the RAG pipeline over the provided queries and collect answers/contexts.

    Parameters:
        df: Input DataFrame must contain column "user_input" with user queries.

    Returns:
        A tuple (out_df, llm, embedder) where:
            - out_df: df with two added columns:
                - "answer": model-generated answer string per query.
                - "contexts": list[str] per query containing retrieved context chunks.
            - llm: identifier/name of the LLM used by the pipeline.
            - embedder: identifier/name of the embedding model used in retrieval.
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