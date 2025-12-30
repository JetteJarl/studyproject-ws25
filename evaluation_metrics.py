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

from evaluation_wrappers import MistralRagasLLM, RagasHuggingFaceWrapper
from evaluation_dataset import load_fever_split, run_pipeline_on_querys
from llm_model import MistralModel

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
    # llm = LocalOllamaRagasLLM(llm, base_url="http://localhost:11434")
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
    return per_row


def main():
    """
    Orchestrate a full evaluation run:
      1) Load and prepare the FEVER subset.
      2) Run the RAG pipeline to produce answers and contexts.
      3) Evaluate outputs with Ragas and save results.
    """
    df = load_fever_split(sample_size=10)
    llm = "open-mixtral-8x7b"
    chain = MistralModel(llm)
    df, llm, embedder = run_pipeline_on_querys(df, chain, llm)
    per_row = evaluate_with_ragas(df, llm, embedder)

    # Save detailed results
    per_row.to_csv("eval/ragas_fever_per_row_mistral.csv", index=False)
    print("Saved metrics results")


if __name__ == "__main__":
    main()