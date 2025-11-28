# retrieval: (context) precision(@k), (context) recall(@k), mean reciprocal rank (MRR), normalized discounted cumulative gain (NDCG), hit rate, noise sensitivity
# generation: faithfulness, answer relevancy, context relevancy, correctness, groundedness, completeness, hallucination rate, structure, coherence/clarity, conciseness
# system: latency, source quality, citation quality, toxicity/safety, robustness
# human: helpfulness, truthfulness, persuasiveness, bias/neutrality, tone, readability, structure
# methods: evaluation dataset, LLM as a judge, human evaluation

from typing import Any

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

def evaluate_with_ragas(df: pd.DataFrame, llm: str, embedder: str) -> tuple[Any, list[dict[str, Any]]]:
    """
    Runs a selection of Ragas metrics aligned to your evaluation plan.
    Returns (per-row scores DataFrame, aggregate scores dict).
    """
    # Ragas accepts a pandas DataFrame with columns:
    # query (str), answer (str), contexts (List[str]), ground_truth (str)
    eval_df = df[["query", "answer", "contexts", "ground_truth"]]
    print(f"Data type of the first value in 'query': {type(eval_df['query'].iloc[0])}")
    print(f"Data type of the first value in 'answer': {type(eval_df['answer'].iloc[0])}")
    print(f"Data type of the first value in 'contexts': {type(eval_df['contexts'].iloc[0]), type(eval_df['contexts'].iloc[0][0])}")
    print(f"Data type of the first value in 'ground_truth': {type(eval_df['ground_truth'].iloc[0])}")

    print(eval_df["query"])
    print(eval_df["answer"])
    print(eval_df["contexts"])
    print(eval_df["ground_truth"])

    # Required by retrieval metrics (ContextPrecision, ContextRecall, etc.)
    eval_df["user_input"] = eval_df["query"]

    ragas_dataset = Dataset.from_pandas(eval_df)

    print("type of llm: ", type(llm))
    llm = LocalOllamaRagasLLM(
        model=llm,
        base_url="http://localhost:11434"
    )
    print("type of llm: ", type(llm))

    print("type of embedder: ", type(embedder))
    embedder = RagasHuggingFaceWrapper(model_name=embedder)
    print("type of embedder: ", type(embedder))

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

    result = evaluate(ragas_dataset, metrics=selected_metrics)
    # result.scores is a dict of aggregate metric name -> score
    # result.raw_results is a per-row DataFrame
    per_row = result.to_pandas()  # per-row metric values
    aggregates = result.scores  # dict of aggregate scores
    return per_row, aggregates


def main():
    df = load_fever_split(sample_size=10)
    df, llm, embedder = run_pipeline_on_querys(df)
    per_row, aggregates = evaluate_with_ragas(df, llm, embedder)

    print("Aggregate metrics:")
    for a in aggregates:
        for k, v in a.items():
            print(f"- {k}: {v:.4f}")

    # Save detailed results
    per_row.to_csv("ragas_fever_per_row.csv", index=False)
    pd.DataFrame([aggregates]).to_csv("ragas_fever_aggregates.csv", index=False)
    print("Saved ragas_fever_per_row.csv and ragas_fever_aggregates.csv")


if __name__ == "__main__":
    main()