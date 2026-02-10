# Retrieval: (context) precision(@k), (context) recall(@k), mean reciprocal rank (MRR), normalized discounted cumulative gain (NDCG), hit rate, noise sensitivity
# Generation: faithfulness, answer relevancy, context relevancy, correctness, groundedness, completeness, hallucination rate, structure, coherence/clarity, conciseness
# System: latency, source quality, citation quality, toxicity/safety, robustness
# Human: helpfulness, truthfulness, persuasiveness, bias/neutrality, tone, readability, structure
# Methods: evaluation dataset, LLM as a judge, human evaluation
import csv
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
# Note: not all metrics mentioned above are available in ragas. Especially these subjective metrics like e.g. bias
# Retrieval metrics
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    NoiseSensitivity
)
# Generation metrics
from ragas.metrics import (
    Faithfulness,
    ResponseGroundedness,
    AnswerAccuracy,
    FactualCorrectness,
    AnswerRelevancy,
    ContextRelevance,
    SemanticSimilarity
)

from evaluation_wrappers import MistralRagasLLM, RagasHuggingFaceWrapper
from evaluation_dataset import load_climate_fever_refutes_split, run_pipeline_on_querys
from llm_model import MistralModel

from vector_database import load_vectorstore, get_retriever
from embed_model import get_embeddings_model

def _append_average_row(per_row: pd.DataFrame) -> pd.DataFrame:
    """
    Append a final summary row containing the mean of each numeric metric column.

    This helper is intended for evaluation tables where each row corresponds to a
    single sample (claim) and each column corresponds to a metric score. It computes
    the column-wise arithmetic mean over all *numeric* columns and appends the result
    as a new last row. This makes it easy to inspect overall performance in the
    exported CSV without calculating averages separately.

    Args:
        per_row:
            DataFrame containing per-sample metric outputs (typically produced by
            ``ragas.evaluate(...).to_pandas()``). Expected to have one row per sample
            and one column per metric.

    Returns:
        A new DataFrame with the same columns as ``per_row`` plus (optionally) a
        ``"__row_type__"`` column, and with one extra final row containing the
        averaged metric values.

    Notes:
        - If ``per_row`` is empty, the appended row will contain blanks for all
          columns because there are no values to average.
        - If a numeric column contains NaNs, pandas will ignore them when computing
          the mean (default behavior).
    """
    out = per_row.copy()

    # Compute means for numeric metric columns only
    means = out.mean(numeric_only=True)

    # Build the summary row with the same columns as 'out'
    avg_row = {col: "" for col in out.columns}
    for col, val in means.items():
        avg_row[col] = float(val) if pd.notna(val) else ""

    # Add a row-type marker so the last row is clearly identifiable in the CSV
    if "__row_type__" not in out.columns:
        out["__row_type__"] = "row"
        avg_row["__row_type__"] = "average"
    else:
        avg_row["__row_type__"] = "average"

    out = pd.concat([out, pd.DataFrame([avg_row])], ignore_index=True)
    return out

def evaluate_with_ragas(df: pd.DataFrame, llm: str, embedder: str) -> pd.DataFrame:
    """
    Run Ragas metrics on the model outputs using the specified LLM and embedder backends.

    Parameters:
        df: DataFrame with columns {"user_input", "answer", "contexts", "ground_truth"}.
        llm: Model identifier/name for the judge LLM backend.
        embedder: Embedding model identifier/name.

    Returns:
        pandas DataFrame of per-sample metric scores.
    """
    # Ragas accepts a pandas DataFrame with columns:
    # user_input (str), answer (str), contexts (List[str]), ground_truth (str)
    # Ragas exactly expects these column names and nothing else!
    eval_df = df[["user_input", "answer", "contexts", "ground_truth"]]
    ragas_dataset = Dataset.from_pandas(eval_df)

    # Wrap the LLM and embedder backends with Ragas-specific wrappers
    llm = MistralRagasLLM(llm)
    embedder = RagasHuggingFaceWrapper(embedder)

    # Naming of parameters is here actually necessary
    selected_metrics = [
        # Retrieval metrics
        ContextPrecision(llm=llm), # Measures how many of the retrieved context passages were actually relevant to the answer.
        ContextRecall(llm=llm), # Measures how many relevant context passages the retriever successfully found out of all possible relevant ones.
        NoiseSensitivity(llm=llm), # Measures how much irrelevant or noisy context negatively affects the model’s answer.
        ContextEntityRecall(llm=llm), # Checks whether the retrieved context contains all important entities needed to answer the query correctly.
        # Generation metrics
        Faithfulness(llm=llm), # Evaluates whether the answer is grounded in the retrieved context without hallucinating new facts.
        ResponseGroundedness(llm=llm), # Measures how well each part of the answer can be directly supported by the provided context passages.
        AnswerAccuracy(llm=llm), # Compares the answer directly to the ground-truth answer to measure factual correctness.
        FactualCorrectness(llm=llm), # Checks whether the answer's factual claims are correct relative to the ground truth (precision/recall/f1).
        AnswerRelevancy(llm=llm, embeddings=embedder), # Measures how relevant the generated answer is to the user’s original question.
        ContextRelevance(llm=llm), # Checks whether the retrieved context is actually helpful for answering the question.
        SemanticSimilarity(embeddings=embedder) # Measures how semantically similar the generated answer is to the ground-truth answer.
    ]

    result = evaluate(ragas_dataset, selected_metrics)
    per_row = result.to_pandas()  # per-row metric values

    # Append a final row with the mean of each numeric metric column
    per_row = _append_average_row(per_row)
    return per_row


def main():
    """
    Orchestrate a full evaluation run:
      1) Load and prepare the dataset.
      2) Run the RAG pipeline to produce answers and contexts.
      3) Evaluate outputs with Ragas and save results.
    """
    df = load_climate_fever_refutes_split(sample_size=1)
    llm_name = "open-mixtral-8x7b" # TODO: should enter name via cli and not hardcode!
    llm = MistralModel(llm_name)
    embedder = "sentence-transformers/all-mpnet-base-v2"
    embedding_model, embedder  = get_embeddings_model(embedder)
    number_relevant_chunks = 3

    # Attempt to load an existing vectorstore; if not found, ingest from the URL
    vectorstore = load_vectorstore(embedding_model, "chroma_db")
    if vectorstore is None:
        print("There is no data store available. Look at the .README for more insights or contact the GitHub contributors.")

    # Configure retriever (k controls number of top documents to fetch)
    retriever = get_retriever(vectorstore, number_relevant_chunks)

    df, llm, embedding_model = run_pipeline_on_querys(
        df, llm, embedding_model, retriever
    )
    # evaluate_with_ragas expects model identifiers (strings), not instantiated objects.
    # Pass the original llm_name and embedder identifier.
    per_row = evaluate_with_ragas(df, llm_name, embedder)

    # Save detailed results
    Path("eval").mkdir(parents=True, exist_ok=True)
    per_row.to_csv(
        "eval/ragas_climate_fever_mistral.csv",
        index=False,
        header=True,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )
    print("Saved metrics results")


if __name__ == "__main__":
    main()