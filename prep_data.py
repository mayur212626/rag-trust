"""
prep_data.py

Builds:
  corpus.jsonl:    {doc_id, text}, one per Wikipedia paragraph
  questions.jsonl: {id, question, gold_answer, hard_case}
  heldout.jsonl:   10 unseen questions for qualitative analysis

Source: HotpotQA dev (distractor setting), Yang et al. 2018, EMNLP.
arXiv 1809.09600. Verified via ACL Anthology https://aclanthology.org/D18-1259/

Usage:
    pip install datasets
    python prep_data.py
"""

import json
import random
from datasets import load_dataset

random.seed(42)
N_EVAL = 50
N_HELDOUT = 10


def main():
    ds = load_dataset("hotpot_qa", "distractor", split="validation",
                      trust_remote_code=True)
    shuffled = ds.shuffle(seed=42)
    eval_split = shuffled.select(range(N_EVAL))
    held_split = shuffled.select(range(N_EVAL, N_EVAL + N_HELDOUT))

    corpus = {}
    eval_qs, held_qs = [], []

    def add_example(ex, idx, prefix):
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            doc_id = title.replace(" ", "_")[:60]
            text = " ".join(sents).strip()
            if doc_id not in corpus:
                corpus[doc_id] = text
        return {
            "id": f"{prefix}_{idx:03d}",
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "hard_case": False,
        }

    for i, ex in enumerate(eval_split):
        eval_qs.append(add_example(ex, i, "hp"))
    for i, ex in enumerate(held_split):
        held_qs.append(add_example(ex, i, "ho"))

    # 5 pre-registered hard cases. Replace gold answers if your final corpus
    # cannot answer them; keep the type of failure mode the same.
    hard = [
        {"id": "hard_01_numeric", "hard_case": True,
         "question": "What is the difference in population between the largest cities of Australia and New Zealand?",
         "gold_answer": "approx 3.6 million (verify against retrieved corpus)"},
        {"id": "hard_02_negation", "hard_case": True,
         "question": "Which of the following is NOT a noble gas: argon, nitrogen, neon, krypton?",
         "gold_answer": "nitrogen"},
        {"id": "hard_03_entity_collision", "hard_case": True,
         "question": "What university does the statistician Michael I. Jordan teach at?",
         "gold_answer": "UC Berkeley"},
        {"id": "hard_04_temporal", "hard_case": True,
         "question": "Who is the current CEO of Twitter?",
         "gold_answer": "ambiguous-by-design; report whether system flags staleness"},
        {"id": "hard_05_aggregation", "hard_case": True,
         "question": "List every winner of the Turing Award from 2015 through 2018.",
         "gold_answer": "Whitfield Diffie & Martin Hellman (2015); Tim Berners-Lee (2016); John Hennessy & David Patterson (2017); Yoshua Bengio, Geoffrey Hinton & Yann LeCun (2018)"},
    ]
    eval_qs.extend(hard)

    with open("corpus.jsonl", "w") as f:
        for doc_id, text in corpus.items():
            f.write(json.dumps({"doc_id": doc_id, "text": text}) + "\n")
    with open("questions.jsonl", "w") as f:
        for q in eval_qs:
            f.write(json.dumps(q) + "\n")
    with open("heldout.jsonl", "w") as f:
        for q in held_qs:
            f.write(json.dumps(q) + "\n")

    n_hard = sum(1 for q in eval_qs if q["hard_case"])
    print(f"Wrote corpus.jsonl ({len(corpus)} chunks)")
    print(f"Wrote questions.jsonl ({len(eval_qs)} questions, {n_hard} hard)")
    print(f"Wrote heldout.jsonl ({len(held_qs)} questions)")


if __name__ == "__main__":
    main()
