"""
rescore_partials.py

Re-scores every PARTIAL verdict from results.json using combined evidence.

For single-chunk PARTIALs: re-judges with the same chunk (consistency check).
For multi-chunk PARTIALs: combines ALL cited chunks into one evidence block
and re-judges. This tests the professor's hypothesis: are PARTIALs just
multi-hop reasoning that per-chunk scoring can't represent?

Usage:
    python rescore_partials.py --results results.json --out rescore_results.json

Cost: roughly $0.50-1.00 (55 re-judge calls with claude-opus-4-7).
"""

import os
import json
import argparse
from collections import defaultdict
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")

# Single-chunk judge prompt (same as original)
SINGLE_SYS = (
    "You are a strict faithfulness judge. Given a claim and a source chunk, decide whether the "
    "source LITERALLY supports the claim. Be strict: paraphrases that change scope, time, or "
    "magnitude are NOT supported. A claim with extra information beyond what the source says "
    "is NOT supported. Respond with exactly one of: SUPPORTED, PARTIAL, UNSUPPORTED, "
    "CONTRADICTED. No explanation."
)

# Combined-chunk judge prompt (the professor's suggestion)
COMBINED_SYS = (
    "You are a strict faithfulness judge. Given a claim and multiple source chunks, decide whether "
    "the sources TOGETHER literally support the claim through valid multi-hop reasoning. "
    "The claim is SUPPORTED if the chain of evidence across the sources justifies it without "
    "requiring invented steps. It is PARTIAL if the sources get close but leave a gap. "
    "It is UNSUPPORTED if the sources do not address the claim. "
    "It is CONTRADICTED if the sources say something opposite. "
    "Respond with exactly one of: SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED. No explanation."
)


def parse_verdict(text: str) -> str:
    text = text.strip().upper()
    for tag in ("CONTRADICTED", "UNSUPPORTED", "PARTIAL", "SUPPORTED"):
        if tag in text:
            return tag
    return "UNSUPPORTED"


def judge_single(claim: str, doc_id: str, chunk: str) -> str:
    user = f"CLAIM: {claim}\n\nSOURCE [{doc_id}]:\n{chunk}\n\nVerdict (one word):"
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=10, system=SINGLE_SYS,
        messages=[{"role": "user", "content": user}],
    )
    return parse_verdict(resp.content[0].text)


def judge_combined(claim: str, chunks: list) -> str:
    """chunks = list of (doc_id, text) tuples"""
    evidence = "\n\n".join(f"SOURCE [{doc_id}]:\n{text}" for doc_id, text in chunks)
    user = f"CLAIM: {claim}\n\n{evidence}\n\nVerdict (one word):"
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=10, system=COMBINED_SYS,
        messages=[{"role": "user", "content": user}],
    )
    return parse_verdict(resp.content[0].text)


def extract_partials(results: list) -> list:
    """
    Extract unique (claim, doc_ids, chunks) groups where at least one verdict was PARTIAL.
    Deduplicates by (claim, sorted doc_ids) so we do not re-judge the same pair twice.
    """
    seen = set()
    partial_groups = []

    for r in results:
        for mode in ["baseline", "cove", "quote"]:
            retrieved = r["modes"][mode]["result"]["retrieved"]
            chunk_lookup = {c["doc_id"]: c["text"] for c in retrieved}

            verdicts = r["modes"][mode]["score"].get("verdicts", [])
            if not verdicts:
                continue

            claim_groups = defaultdict(list)
            for v in verdicts:
                claim_groups[v["claim"]].append(v)

            for claim, vs in claim_groups.items():
                if not any(v["verdict"] == "PARTIAL" for v in vs):
                    continue
                doc_ids = [v["doc_id"] for v in vs]
                key = (claim, tuple(sorted(doc_ids)))
                if key in seen:
                    continue
                seen.add(key)

                chunks = [(did, chunk_lookup.get(did, "")) for did in doc_ids]
                partial_groups.append({
                    "qid": r["qid"],
                    "question": r["question"],
                    "mode": mode,
                    "claim": claim,
                    "original_verdicts": {v["doc_id"]: v["verdict"] for v in vs},
                    "doc_ids": doc_ids,
                    "chunks": chunks,
                    "is_multi_chunk": len(doc_ids) > 1,
                })

    return partial_groups


def rescore(results_path: str, out_path: str):
    results = json.load(open(results_path))
    groups = extract_partials(results)

    print(f"Found {len(groups)} unique PARTIAL claim groups to re-score.")
    multi = [g for g in groups if g["is_multi_chunk"]]
    single = [g for g in groups if not g["is_multi_chunk"]]
    print(f"  Multi-chunk (combined evidence test): {len(multi)}")
    print(f"  Single-chunk (consistency check): {len(single)}")
    print()

    rows = []
    flipped_to_supported = 0
    stayed_partial = 0
    went_unsupported = 0

    for i, g in enumerate(groups):
        claim = g["claim"]
        chunks = g["chunks"]

        combined_verdict = judge_combined(claim, chunks)

        if not g["is_multi_chunk"]:
            single_verdict = judge_single(claim, chunks[0][0], chunks[0][1])
        else:
            single_verdict = None

        if combined_verdict == "SUPPORTED":
            flipped_to_supported += 1
        elif combined_verdict == "PARTIAL":
            stayed_partial += 1
        else:
            went_unsupported += 1

        row = {
            "qid": g["qid"],
            "question": g["question"],
            "mode": g["mode"],
            "claim": claim,
            "doc_ids": g["doc_ids"],
            "is_multi_chunk": g["is_multi_chunk"],
            "original_verdicts": g["original_verdicts"],
            "combined_verdict": combined_verdict,
            "single_verdict": single_verdict,
            "flipped": combined_verdict == "SUPPORTED",
        }
        rows.append(row)

        flip_marker = "FLIPPED TO SUPPORTED" if combined_verdict == "SUPPORTED" else combined_verdict
        print(f"[{i+1}/{len(groups)}] {g['qid']} | multi={g['is_multi_chunk']} | {flip_marker}")
        print(f"  Claim: {claim[:80]}")
        print()

    total = len(rows)
    multi_rows = [r for r in rows if r["is_multi_chunk"]]
    single_rows = [r for r in rows if not r["is_multi_chunk"]]
    multi_flipped = sum(1 for r in multi_rows if r["flipped"])
    single_flipped = sum(1 for r in single_rows if r["flipped"])

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total PARTIAL groups re-scored: {total}")
    print(f"  Flipped to SUPPORTED: {flipped_to_supported} ({flipped_to_supported/total:.0%})")
    print(f"  Stayed PARTIAL:       {stayed_partial} ({stayed_partial/total:.0%})")
    print(f"  Went UNSUPPORTED:     {went_unsupported} ({went_unsupported/total:.0%})")
    print()
    if multi_rows:
        print(f"Multi-chunk groups ({len(multi_rows)} total):")
        print(f"  Flipped to SUPPORTED: {multi_flipped} ({multi_flipped/len(multi_rows):.0%})")
    if single_rows:
        print(f"Single-chunk groups ({len(single_rows)} total):")
        print(f"  Flipped to SUPPORTED: {single_flipped} ({single_flipped/len(single_rows):.0%})")
    print()

    if multi_rows and multi_flipped / len(multi_rows) > 0.5:
        conclusion = (
            "Most multi-chunk PARTIALs flipped to SUPPORTED when chunks were combined. "
            "This supports the professor's hypothesis: the dominant failure mode is a metric "
            "limitation, not genuine model overclaiming. Per-chunk scoring cannot represent "
            "multi-hop grounding."
        )
    else:
        conclusion = (
            "Most PARTIALs did not flip even with combined evidence. "
            "This supports the original conclusion: the model genuinely overclaims "
            "beyond what the combined sources support."
        )
    print("INTERPRETATION:", conclusion)

    output = {
        "summary": {
            "total_rescored": total,
            "flipped_to_supported": flipped_to_supported,
            "stayed_partial": stayed_partial,
            "went_unsupported": went_unsupported,
            "flip_rate_overall": round(flipped_to_supported / total, 4) if total else 0,
            "multi_chunk": {
                "n": len(multi_rows),
                "flipped": multi_flipped,
                "flip_rate": round(multi_flipped / len(multi_rows), 4) if multi_rows else 0,
            },
            "single_chunk": {
                "n": len(single_rows),
                "flipped": single_flipped,
                "flip_rate": round(single_flipped / len(single_rows), 4) if single_rows else 0,
            },
            "interpretation": conclusion,
        },
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="rescore_results.json")
    args = ap.parse_args()
    rescore(args.results, args.out)
