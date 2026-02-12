"""
Purpose
-------
Create an evaluation-ready JSONL file from Climate-FEVER where:
  - We keep ONLY claims with claim_label == "REFUTES"
  - We build a "ground_truth" reference text by concatenating the top-k
    REFUTES-labeled evidence sentences for that claim
  - We DROP any REFUTES claim that has zero REFUTES evidence sentences

Why this is useful
------------------
For RAG evaluation, you often need:
  - user_input: the claim to respond to
  - ground_truth: a reference text to compare against (e.g., for similarity/correctness metrics)

This script converts Climate-FEVER into that format.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def build_ground_truth_from_evidences(
    evidences: List[Dict[str, Any]],
    include_article_prefix: bool,
    top_k: int,
) -> str:
    """
    Build a ground-truth reference string from the evidence list of a single claim.

    Steps:
      1) Keep only evidence items where evidence_label == "REFUTES"
         (because we are evaluating refuted claims and want refuting evidence).
      2) Sort by entropy ascending:
           - lower entropy = annotators agree more
           - higher entropy = more disagreement/noise
      3) Take the top_k best evidence sentences after sorting.
      4) Concatenate them into one text block, optionally prefixed by article title.

    Returns:
      A single string (possibly multi-line). Empty string means we found no usable evidence.
    """
    # Filter to evidence sentences that explicitly refute the claim
    refuting = [e for e in evidences if e.get("evidence_label") == "REFUTES"]

    # Sort by lowest entropy first (most agreement), then by evidence_id for deterministic ordering
    refuting.sort(
        key=lambda e: (float(e.get("entropy", 1e9)), str(e.get("evidence_id", "")))
    )

    # Select top-k; if top_k is 0, we interpret it as "keep all"
    selected = refuting if top_k == 0 else refuting[: max(top_k, 0)]

    parts: List[str] = []
    for e in selected:
        # Extract the actual evidence sentence text
        sentence = str(e.get("evidence", "")).strip()
        if not sentence:
            # Skip empty evidence strings
            continue

        # Optionally add article title prefix (helps give a hint of provenance)
        if include_article_prefix:
            article = str(e.get("article", "")).strip()
            parts.append(f"{article}: {sentence}" if article else sentence)
        else:
            parts.append(sentence)

    # Join as a multi-line reference text
    return "\n".join(parts).strip()


def process_file(
    input_path: Path,
    output_path: Path,
    include_article_prefix: bool = True,
    top_k: int = 3,
) -> Tuple[int, int, int]:
    """
    Read Climate-FEVER JSONL from input_path and write filtered JSONL to output_path.

    Filtering logic:
      - Only keep rows where claim_label == "REFUTES"
      - Build ground_truth from REFUTES evidences because these are refuting incorrect claims
      - Drop the row if ground_truth ends up empty (no refuting evidence)

    Output format (one JSON per line):
      {
        "claim_id": "...",
        "user_input": "...",      # the claim text
        "ground_truth": "...",    # concatenated top-k refuting evidence sentences
        "claim_label": "REFUTES"
      }

    Returns:
      (total_rows_read, refutes_claims_found, refutes_claims_kept)
    """
    total = 0
    refutes_total = 0
    kept = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                # Skip blank lines
                continue

            total += 1
            row = json.loads(line)

            # Keep only REFUTES claims
            if row.get("claim_label") != "REFUTES":
                continue
            refutes_total += 1

            # Evidences should be a list; handle missing/invalid shapes safely
            evidences = row.get("evidences") or []
            if not isinstance(evidences, list):
                evidences = []

            # Build ground_truth from the claim's REFUTES evidence sentences
            ground_truth = build_ground_truth_from_evidences(
                evidences,
                top_k=top_k,
                include_article_prefix=include_article_prefix,
            )

            # Drop REFUTES claims that don't have any usable REFUTES evidence
            if not ground_truth:
                continue

            # Create the output row in a RAG-eval-friendly schema
            out_row = {
                "claim_id": str(row.get("claim_id", "")).strip(),
                "user_input": str(row.get("claim", "")).strip(),
                "ground_truth": ground_truth,
                "claim_label": "REFUTES",
            }

            # Write as JSONL (one JSON object per line)
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1

    return total, refutes_total, kept


def main() -> None:
    """
    CLI entry point.

    You can control:
      - input file path
      - output file path
      - top-k evidence sentences to include
      - whether to prefix each evidence sentence with its article title
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract REFUTES claims from Climate-FEVER and build ground_truth from REFUTES evidences. "
            "Drops claims with no REFUTES evidence."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/climate-fever.jsonl"),
        help="Path to climate-fever.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/climate-fever-refutes-groundtruth.jsonl"),
        help="Path to write the filtered JSONL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "Number of refuting evidence sentences to include in ground_truth "
            "(sorted by lowest entropy first). Use 0 to include all."
        ),
    )
    parser.add_argument(
        "--include-article-prefix",
        action="store_true",
        help="Prefix each evidence sentence with '<article>: ' in ground_truth.",
    )

    args = parser.parse_args()

    # Basic validation: fail early if the input file does not exist
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run the transformation
    total, refutes_total, kept = process_file(
        args.input,
        args.output,
        top_k=args.top_k,
        include_article_prefix=args.include_article_prefix,
    )

    # Print summary stats so you know how much data you kept/dropped
    dropped = refutes_total - kept
    print(f"Read total rows:                 {total}")
    print(f"REFUTES claims found:            {refutes_total}")
    print(f"Kept (has REFUTES evidence):     {kept}")
    print(f"Dropped (no REFUTES evidence):   {dropped}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()