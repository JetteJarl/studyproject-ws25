# retrieval: (context) precision(@k), (context) recall(@k), mean reciprocal rank (MRR), normalized discounted cumulative gain (NDCG), hit rate, noise sensitivity
# generation: faithfulness, answer relevancy, context relevancy, correctness, groundedness, completeness, hallucination rate, structure, coherence/clarity, conciseness
# system: latency, source quality, citation quality, toxicity/safety, robustness
# human: helpfulness, truthfulness, persuasiveness, bias/neutrality, tone, readability, structure
# methods: evaluation dataset, LLM as a judge, human evaluation

from typing import List, Any

import pandas as pd
from datasets import Dataset
from langchain_ollama import OllamaLLM
from ragas.embeddings import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.llms.base import LangchainLLMWrapper
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
    # ResponseGroundedness,
    # AnswerAccuracy,
    FactualCorrectness,
    AnswerRelevancy,
    # ContextRelevance,
    SemanticSimilarity
)

from llm_model import generate_answer
from rag_pipeline import load_rag

class LocalOllamaRagasLLM(LangchainLLMWrapper):
    def __init__(self, model: str, base_url: str):
        self.langchain_llm = OllamaLLM(model=model, base_url=base_url)
        self.bypass_temperature = True
        self.is_finished_parser = None

    async def _agenerate(self, prompt: str) -> str:
        return await self.langchain_llm.apredict(prompt)

    def set_run_config(self, config):
        self.run_config = config

def build_ground_truth_text(evidences: List[List[Any]]) -> str:
    """
    FEVER provides evidence as lists of [annotator_id, page, sentence_idx].
    Some dataset variants include the sentence text, others only ids.
    This function assumes you have text in the processed record. If not, it will
    fall back to a short label-based string, which still enables 'correctness'.
    """
    # If you do not have evidence sentence texts, you can just return an empty string
    # and rely on 'correctness' built from labels, or craft a short reference.
    # Here we join any available evidence_text fields if present.
    parts: List[str] = []
    for ev_group in evidences or []:
        for ev in ev_group:
            # Try common keys if your preprocessing attaches text
            if isinstance(ev, dict) and "text" in ev:
                parts.append(str(ev["text"]))
            elif isinstance(ev, (list, tuple)) and len(ev) >= 4:
                # Example layout if you pre-augment: [ann_id, page, sent_idx, text]
                parts.append(str(ev[3]))
    return " ".join(parts).strip()


def load_fever_split(sample_size: int = 200, seed: int = 7) -> pd.DataFrame:
    """
    Load a FEVER split and prepare a DataFrame with columns:
    - query: the claim
    - ground_truth: evidence text or label-based reference
    - label: SUPPORTS/REFUTES/NOT ENOUGH INFO
    We filter out NOT ENOUGH INFO for clean supervision.
    """
    # Newer versions of dataset 3.6.0 contain a bug that prevents loading datasets remotely:
    # https://github.com/huggingface/datasets/issues/7693
    # Downloaded the data manually from https://fever.ai/dataset/fever.html and save it locally
    df = pd.read_json("data/shared_task_dev.jsonl", lines=True)
    print(df.head)
    print(df.columns)
    print(df.shape)
    # Keep only entries with label != NEI
    df = df[df["label"].isin(["SUPPORTS", "REFUTES"])].copy()

    # Build a lightweight reference text. If you can’t extract evidence sentences,
    # at least encode the label in the ground truth to help 'correctness'.
    def _ref_text(row) -> str:
        evidence = f"The claim is {row['label'].lower()} according to Wikipedia evidence."
        return evidence

    df["query"] = df["claim"].astype(str)
    df["ground_truth"] = df.apply(_ref_text, axis=1)

    # Evaluate on a very small subset for speed (e.g., 10 rows)
    df = df.head(10).reset_index(drop=True)
    # Sample for a quick evaluation run
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)

    # Keep only necessary columns
    return df[["query", "ground_truth", "label"]].reset_index(drop=True)


def run_pipeline_on_querys(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """
    Calls your RAG pipeline for each query.
    Expects load_rag(query) -> (answer: str, contexts: List[str], llm: str, embedder: str).
    """
    answers: List[str] = []
    contexts_list: List[List[str]] = []

    retriever, chain, llm, embedder = load_rag()

    for query in df["query"].tolist():
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
    embedder = HuggingFaceEmbeddings(model=embedder)
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
        # ResponseGroundedness(llm=llm),
        # AnswerAccuracy(llm=llm),
        FactualCorrectness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embedder),
        # ContextRelevance(llm=llm),
        SemanticSimilarity(embeddings=embedder)
    ]

    result = evaluate(ragas_dataset, metrics=selected_metrics)
    # result.scores is a dict of aggregate metric name -> score
    # result.raw_results is a per-row DataFrame
    per_row = result.to_pandas()  # per-row metric values
    aggregates = result.scores  # dict of aggregate scores
    return per_row, aggregates


def main():
    df = load_fever_split(sample_size=200)
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