"""
judge.py

LLM-as-judge for citation faithfulness.
Also exports a hand-label/auto-label agreement utility (Cohen's kappa)
which the pre-registration requires before the judge prompt is locked.
"""

import os
import json
from typing import List, Dict
from anthropic import Anthropic
from rag_pipeline import RAGResult, segment_claims

_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")

JUDGE_SYS = (
    "You are a strict faithfulness judge. Given a claim and a source chunk, decide whether the "
    "source LITERALLY supports the claim. Be strict: paraphrases that change scope, time, or "
    "magnitude are NOT supported. A claim with extra information beyond what the source says "
    "is NOT supported. Respond with exactly one of: SUPPORTED, PARTIAL, UNSUPPORTED, "
    "CONTRADICTED. No explanation."
)

JUDGE_TMPL = "CLAIM: {claim}\n\nSOURCE [{doc_id}]:\n{chunk}\n\nVerdict (one word):"


def judge_pair(claim: str, doc_id: str, chunk_text: str, model: str = None) -> str:
    model = model or JUDGE_MODEL
    resp = _client.messages.create(
        model=model,
        max_tokens=10,
        system=JUDGE_SYS,
        messages=[{"role": "user", "content": JUDGE_TMPL.format(
            claim=claim, doc_id=doc_id, chunk=chunk_text)}],
    )
    out = resp.content[0].text.strip().upper()
    for tag in ("CONTRADICTED", "UNSUPPORTED", "PARTIAL", "SUPPORTED"):
        if tag in out:
            return tag
    return "UNSUPPORTED"


def score_result(result: RAGResult) -> Dict:
    """Score a single RAGResult. Returns CSR, citation precision, verdicts."""
    if result.abstained:
        return {"csr": None, "citation_precision": None, "abstained": True,
                "n_citations": 0, "verdicts": []}

    retrieved_ids = {c.doc_id for c in result.retrieved}
    chunk_lookup = {c.doc_id: c.text for c in result.retrieved}

    claims = segment_claims(result.answer)
    verdicts = []
    cited_in_retrieved, total_cites = 0, 0

    for item in claims:
        for cid in item["cited_ids"]:
            total_cites += 1
            if cid in retrieved_ids:
                cited_in_retrieved += 1
                v = judge_pair(item["claim"], cid, chunk_lookup[cid])
            else:
                v = "UNSUPPORTED"  # hallucinated doc id
            verdicts.append({"claim": item["claim"], "doc_id": cid, "verdict": v})

    if not verdicts:
        return {"csr": None, "citation_precision": None, "abstained": False,
                "n_citations": 0, "verdicts": []}

    supported = sum(1 for v in verdicts if v["verdict"] == "SUPPORTED")
    return {
        "csr": supported / len(verdicts),
        "citation_precision": cited_in_retrieved / total_cites if total_cites else 0.0,
        "abstained": False,
        "n_citations": len(verdicts),
        "verdicts": verdicts,
    }


# Judge agreement utility (run BEFORE main eval)

def cohens_kappa(rater_a: List[str], rater_b: List[str]) -> float:
    """Cohen's kappa for two raters with categorical labels."""
    assert len(rater_a) == len(rater_b)
    n = len(rater_a)
    labels = sorted(set(rater_a) | set(rater_b))
    obs = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    counts_a = {l: rater_a.count(l) / n for l in labels}
    counts_b = {l: rater_b.count(l) / n for l in labels}
    exp = sum(counts_a[l] * counts_b[l] for l in labels)
    return (obs - exp) / (1 - exp) if (1 - exp) > 0 else 1.0


def run_judge_agreement(pairs_path: str, out_path: str = "judge_agreement.json") -> Dict:
    """
    pairs_path: jsonl with fields {claim, doc_id, chunk, human_label}
    where human_label in {SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED}.
    """
    pairs = [json.loads(l) for l in open(pairs_path)]
    human, machine = [], []
    rows = []
    for p in pairs:
        m = judge_pair(p["claim"], p["doc_id"], p["chunk"])
        h = p["human_label"]
        human.append(h); machine.append(m)
        rows.append({**p, "machine_label": m})
    # binary collapse: SUPPORTED vs not
    h_bin = ["SUP" if x == "SUPPORTED" else "NSUP" for x in human]
    m_bin = ["SUP" if x == "SUPPORTED" else "NSUP" for x in machine]
    out = {
        "n": len(pairs),
        "agreement_4way": sum(1 for h, m in zip(human, machine) if h == m) / len(pairs),
        "kappa_4way": cohens_kappa(human, machine),
        "agreement_binary": sum(1 for h, m in zip(h_bin, m_bin) if h == m) / len(pairs),
        "kappa_binary": cohens_kappa(h_bin, m_bin),
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"4-way: agreement={out['agreement_4way']:.2%}  kappa={out['kappa_4way']:.3f}")
    print(f"Binary: agreement={out['agreement_binary']:.2%}  kappa={out['kappa_binary']:.3f}")
    return out


if __name__ == "__main__":
    import sys
    run_judge_agreement(sys.argv[1])
