"""
run_eval.py

Run baseline plus 3 interventions over the eval set and dump everything
to results.json. Aggregates and statistical analysis live in analysis.py.

Usage:
    python run_eval.py --corpus corpus.jsonl --questions questions.jsonl \\
                       --modes baseline cove nli quote --out results.json
"""

import argparse
import json
from pathlib import Path
from rag_pipeline import RAGPipeline, Chunk, serialize_result
from judge import score_result


def load_corpus(path: str):
    return [Chunk(**json.loads(l)) for l in open(path)]

def load_questions(path: str):
    return [json.loads(l) for l in open(path)]


def run(corpus_path, questions_path, modes, out_path):
    chunks = load_corpus(corpus_path)
    questions = load_questions(questions_path)
    pipes = {m: RAGPipeline(chunks, mode=m) for m in modes}

    rows = []
    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        is_hard = q.get("hard_case", False)
        row = {"qid": qid, "question": question, "hard_case": is_hard,
               "gold_answer": q.get("gold_answer", ""), "modes": {}}
        for m, pipe in pipes.items():
            res = pipe.answer(question)
            score = score_result(res)
            row["modes"][m] = {
                "result": serialize_result(res),
                "score": score,
            }
            print(f"[{i+1}/{len(questions)}] {qid} {m}: csr={score['csr']} "
                  f"abst={res.abstained} lat={res.latency_s:.1f}s "
                  f"toks={res.usage.input_tokens + res.usage.output_tokens}")
        rows.append(row)
        # incremental save in case of crash
        Path(out_path).write_text(json.dumps(rows, indent=2))

    print(f"\nDone. Wrote {out_path} ({len(rows)} rows).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "cove", "nli", "quote"])
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    run(args.corpus, args.questions, args.modes, args.out)
