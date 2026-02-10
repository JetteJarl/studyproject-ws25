from typing import List

import pandas as pd

from llm_model import LlmModel

# Build a lightweight reference text. If you can’t extract evidence sentences,
# at least encode the label in the ground truth to help 'correctness'.
def load_climate_fever_refutes_split(sample_size: int = 100, seed: int = 7) -> pd.DataFrame:
    """
    Load the Climate-FEVER REFUTES-only split (preprocessed) and prepare inputs for evaluation.

    This function reads ``data/climate-fever-refutes-groundtruth.jsonl`` which already contains:
      - user_input: the refuted claim text
      - ground_truth: concatenated REFUTES evidence sentences (top-k)
      - claim_label: "REFUTES"
      - claim_id: identifier

    Parameters:
        sample_size: Maximum number of rows to keep (None or 0 to keep all).
        seed: Random seed used when sampling rows.

    Returns:
        DataFrame with columns:
            - user_input: user claim text (str)
            - ground_truth: reference text built from refuting evidence (str)
            - label: claim label (always "REFUTES" here)
            - claim_id: id string (optional but useful for debugging)
    """
    df = pd.read_json("data/climate-fever-refutes-groundtruth.jsonl", lines=True)

    # Keep only REFUTES rows (should already be true, but keeps it robust)
    df = df[df["claim_label"].astype(str).eq("REFUTES")].copy()

    # Ensure the column names match what the rest of the pipeline expects
    # Ragas expects "user_input" and "ground_truth".
    df["user_input"] = df["user_input"].astype(str)
    df["ground_truth"] = df["ground_truth"].astype(str)

    # Keep a "label" column for compatibility with existing code paths / reporting
    df["label"] = df["claim_label"].astype(str)

    # Sample for a quick evaluation run
    if sample_size and len(df) > sample_size:
        print("Evaluating", sample_size, "samples (out of", len(df), "total samples)...")
        df = df.sample(n=sample_size, random_state=seed)

    # Keep only necessary columns (add claim_id for traceability)
    df = df[["claim_id", "user_input", "ground_truth", "label"]].reset_index(drop=True)

    # Turn claim IDs into integers
    # df["claim_id"] = pd.to_numeric(df["claim_id"], errors="coerce").astype("Int64")
    return df

def _unpack_answer_and_contexts(result: tuple[object, object]) -> tuple[str, object]:
    """
    Validate and unpack the model output.

    The pipeline expects the model to return exactly two items:
    ``(answer, contexts)``.

    Args:
        result: The raw result returned by ``chain.generate_answer(...)``.

    Returns:
        A tuple ``(answer, contexts)`` where:
          - answer: The answer coerced to ``str``.
          - contexts: The raw contexts payload (to be normalized separately).

    Raises:
        ValueError: If the return value is not a 2-tuple.
    """
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            "generate_answer(...) must return a 2-tuple (answer, contexts). "
            f"Got {type(result).__name__} with value: {result!r}"
        )
    answer, contexts = result
    return str(answer), contexts

def _normalize_contexts(contexts: object) -> List[str]:
    """
    Convert a contexts payload into the Ragas-compatible format ``list[str]``.

    Supported input shapes:
        - ``None`` -> []
        - ``str`` -> [str]
        - iterable of strings
        - iterable of Document-like objects (having ``page_content``)

    Args:
        contexts: Raw contexts payload from ``generate_answer``.

    Returns:
        A list of cleaned context passage strings. Each passage is stripped of
        leading/trailing whitespace.
    """
    if contexts is None:
        return []

    if isinstance(contexts, str):
        items = [contexts]
    else:
        items = contexts  # expected iterable

    clean: list[str] = []
    for c in items:
        text = c.page_content if hasattr(c, "page_content") else str(c)
        clean.append(text.strip())
    return clean

def run_pipeline_on_querys(
    df: pd.DataFrame,
    llm: LlmModel,
    llm_name: str,
    embedding_model: any,
    retriever: any,
    number_relevant_chunks: int
) -> tuple[pd.DataFrame, str, str]:
    """
    Run the RAG pipeline for each query in a DataFrame and collect model outputs.

    This function initializes (or loads) the retrieval stack via :func:`rag_pipeline.load_rag`,
    then iterates over ``df["user_input"]`` and calls ``chain.generate_answer(query, retriever)``.
    The outputs are normalized into a format compatible with Ragas evaluation:
    - ``answer`` is stored as a string
    - ``contexts`` is stored as ``list[str]`` (one list per sample)

    Notes:
        - This function relies on two helpers:
            - ``_unpack_answer_and_contexts(...)``: validates/unpacks the model return value.
            - ``_normalize_contexts(...)``: converts contexts into ``list[str]`` by extracting
              ``page_content`` from Document-like objects or coercing items to strings.
        - The expected return shape of ``chain.generate_answer(...)`` is the one supported by
          ``_unpack_answer_and_contexts(...)``. If the model returns an unexpected shape,
          the helper may raise a ``ValueError`` (depending on its implementation).
        - The retrieval configuration (embedding model, vector store, ``k``, etc.) is defined
          inside :func:`rag_pipeline.load_rag`.

    Args:
        df: Input DataFrame that must contain a ``"user_input"`` column with user queries.
            Any additional columns are preserved in the output.
        chain: Model implementation used to generate answers (must implement
            :meth:`llm_model.LlmModel.generate_answer`).
        llm: Identifier/name of the generator model. Passed through to
            :func:`rag_pipeline.load_rag` and returned for logging/reporting.
        embedder: Identifier/name of the embedding model used by the retriever (e.g.
            ``"sentence-transformers/all-mpnet-base-v2"``). Passed through to
            :func:`rag_pipeline.load_rag` and returned for logging/reporting.
        number_relevant_chunks: Number of retrieved context chunks to fetch per query (top-k).
            This controls retriever breadth and the amount of context provided to the model.

    Returns:
        A tuple ``(out_df, llm, embedder)`` where:
            - out_df: Copy of the input DataFrame with two added columns:
                - ``"answer"``: model-generated answer (``str``) per row.
                - ``"contexts"``: normalized context passages (``list[str]``) per row.
            - llm: The resolved generator model identifier/name actually used.
            - embedder: The embedding model identifier/name used by the retriever.

    Raises:
        KeyError: If ``"user_input"`` is missing from the input DataFrame.
        ValueError: If the model output cannot be unpacked by ``_unpack_answer_and_contexts``.
        Exception: Propagates exceptions raised by retriever initialization or model calls.
    """
    answers: List[str] = []
    contexts_list: List[List[str]] = []


    for query in df["user_input"].tolist():
        result = llm.generate_answer(query, retriever)
        answer, contexts = _unpack_answer_and_contexts(result)

        answers.append(answer)
        contexts_list.append(_normalize_contexts(contexts))

    out = df.copy()
    out["answer"] = answers
    out["contexts"] = contexts_list
    return out, llm, embedding_model