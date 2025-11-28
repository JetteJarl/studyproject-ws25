# retrieval: (context) precision(@k), (context) recall(@k), mean reciprocal rank (MRR), normalized discounted cumulative gain (NDCG), hit rate, noise sensitivity
# generation: faithfulness, answer relevancy, context relevancy, correctness, groundedness, completeness, hallucination rate, structure, coherence/clarity, conciseness
# system: latency, source quality, citation quality, toxicity/safety, robustness
# human: helpfulness, truthfulness, persuasiveness, bias/neutrality, tone, readability, structure
# methods: evaluation dataset, LLM as a judge, human evaluation

import pandas as pd
from datasets import Dataset
from ragas import evaluate
# Note: not all metrics mentioned above are available in ragas. Especially these subjective metrics like e.g. bias
# retrieval metrics
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    NoiseSensitivity
)
# generation metrics
from ragas.metrics import (
    Faithfulness,
    ResponseGroundedness,
    AnswerAccuracy,
    FactualCorrectness,
    AnswerRelevancy,
    ContextRelevance,
    SemanticSimilarity
)

from evaluation_wrappers import LocalOllamaRagasLLM, RagasHuggingFaceWrapper
from evaluation_dataset import load_fever_split, run_pipeline_on_querys

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
    eval_df = df[["user_input", "answer", "contexts", "ground_truth"]]
    ragas_dataset = Dataset.from_pandas(eval_df)

    llm = LocalOllamaRagasLLM(llm, base_url="http://localhost:11434")
    embedder = RagasHuggingFaceWrapper(embedder)

    # Naming of parameters is here actually necessary
    selected_metrics = [
        # Retrieval metrics
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        NoiseSensitivity(llm=llm),
        ContextEntityRecall(llm=llm),
        # Generation metrics
        Faithfulness(llm=llm),
        ResponseGroundedness(llm=llm),
        AnswerAccuracy(llm=llm),
        FactualCorrectness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embedder),
        ContextRelevance(llm=llm),
        SemanticSimilarity(embeddings=embedder)
    ]

    result = evaluate(ragas_dataset, selected_metrics)
    per_row = result.to_pandas()  # per-row metric values
    return per_row


def main():
    """
    Orchestrate a full evaluation run:
      1) Load and prepare the FEVER subset.
      2) Run the RAG pipeline to produce answers and contexts.
      3) Evaluate outputs with Ragas and save results.
    """
    df = load_fever_split(sample_size=10)
    df, llm, embedder = run_pipeline_on_querys(df)
    per_row = evaluate_with_ragas(df, llm, embedder)

    # Save detailed results
    per_row.to_csv("ragas_fever_per_row.csv", index=False)
    print("Saved metrics results")


if __name__ == "__main__":
    main()