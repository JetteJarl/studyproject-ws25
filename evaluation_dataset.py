from typing import List, Any

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
    df = df[["user_input", "ground_truth", "label"]].reset_index(drop=True)

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
    Convert a context payload into the Ragas-compatible format ``list[str]``.

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
    embedding_model: Any,
    retriever: Any,
) -> tuple[pd.DataFrame, str, str]:
    """
    Run the RAG pipeline for each query in a DataFrame and collect model outputs.

    For every row in ``df``, this function:
      1) Reads the user query from ``df["user_input"]``.
      2) Calls ``llm.generate_answer(query, retriever)`` to produce:
           - an answer string, and
           - a contexts payload (typically a list of retrieved Document objects).
      3) Normalizes the contexts into the Ragas-compatible shape ``list[str]``.
      4) Returns a copy of the input DataFrame with two new columns:
           - ``answer`` (str)
           - ``contexts`` (List[str])

    The returned DataFrame can then be passed to Ragas evaluation, which expects
    columns named exactly: ``user_input``, ``answer``, ``contexts``, ``ground_truth``.

    Notes:
        - This function relies on two helpers defined in this module:
            - ``_unpack_answer_and_contexts(...)``:
              Ensures ``llm.generate_answer`` returns exactly a 2-tuple
              ``(answer, contexts)`` and coerces the answer to ``str``.
            - ``_normalize_contexts(...)``:
              Converts contexts into ``list[str]``. It supports contexts being:
                * ``None`` (-> [])
                * a single string (-> [string])
                * an iterable of strings
                * an iterable of Document-like objects (must have ``page_content``)

        - The retriever configuration (vector store, top-k, etc.) is not handled
          inside this function; it is assumed to be already constructed and
          passed in via the ``retriever`` argument.

        - ``embedding_model`` is not used directly in this loop. It is accepted
          and returned so the calling code can keep track of which embedding
          backend was used for retrieval (useful for logging/experiment metadata).

    Args:
        df:
            Input DataFrame that must contain a ``"user_input"`` column (str).
            Any additional columns (e.g., ``claim_id``, ``label``) are preserved
            in the returned DataFrame.
        llm:
            The generator model implementation. Must implement
            ``LlmModel.generate_answer(query, retriever)`` and return a tuple
            ``(answer, contexts)``.
        embedding_model:
            The embedding model instance used by the retriever (kept for
            experiment bookkeeping; not used directly here).
        retriever:
            The retrieval backend used to fetch context documents for each query.
            It is passed through to ``llm.generate_answer``.

    Returns:
        tuple[pd.DataFrame, str, str]:
            ``(out_df, llm, embedding_model)`` where:
              - ``out_df`` is a copy of ``df`` with added columns:
                    * ``answer``: model-generated answer (str)
                    * ``contexts``: normalized retrieved contexts (List[str])
              - ``llm`` is the same LLM object that was passed in (returned for
                convenience/consistency with older call sites).
              - ``embedding_model`` is the same embedding model object that was
                passed in (returned for convenience/consistency).

            Note: Even though the return type annotation uses ``str`` for the
            last two items in some older versions of this code, the function
            typically returns the *objects* you pass in (LLM instance and
            embedding model instance).

    Raises:
        KeyError:
            If ``"user_input"`` is missing from the input DataFrame.
        ValueError:
            If ``llm.generate_answer(...)`` does not return a 2-tuple
            ``(answer, contexts)`` (raised by ``_unpack_answer_and_contexts``).
        Exception:
            Propagates any exception raised by the LLM call or retriever.
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