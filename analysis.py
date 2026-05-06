"""
analysis.py

Reads results.json and computes:
  - per-mode CSR with 95% bootstrap CI
  - paired bootstrap p-values (mode vs baseline)
  - citation precision, coverage, abstain rate
  - trust tax: latency, token cost
  - hard-case breakdown
  - faithfulness/cost Pareto plot

Usage:
    pip install matplotlib numpy
    python analysis.py results.json
"""

import json
import sys
import numpy as np
from typing import List, Dict, Optional

N_BOOT = 10_000
RNG = np.random.default_rng(0)


def _csr_per_q(rows, mode):
    """Per-question CSR. Returns list of (csr, n_citations) skipping abstained."""
    out = []
    for r in rows:
        s = r["modes"][mode]["score"]
        if s["abstained"] or s["csr"] is None:
            continue
        out.append((s["csr"], s["n_citations"]))
    return out


def bootstrap_csr(rows, mode):
    pairs = _csr_per_q(rows, mode)
    if not pairs:
        return None
    csrs = np.array([p[0] for p in pairs])
    boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(csrs), len(csrs))
        boots.append(csrs[idx].mean())
    boots = np.array(boots)
    return {"mean": float(csrs.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "n": len(csrs)}


def paired_diff(rows, mode_a, mode_b):
    """Paired bootstrap on questions where BOTH produced citations."""
    diffs = []
    for r in rows:
        sa, sb = r["modes"][mode_a]["score"], r["modes"][mode_b]["score"]
        if sa["abstained"] or sb["abstained"]:
            continue
        if sa["csr"] is None or sb["csr"] is None:
            continue
        diffs.append(sb["csr"] - sa["csr"])
    if not diffs:
        return None
    diffs = np.array(diffs)
    boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(diffs), len(diffs))
        boots.append(diffs[idx].mean())
    boots = np.array(boots)
    p = float((boots <= 0).mean())  # prob improvement is at most 0
    return {"mean_diff": float(diffs.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "p_one_sided": p, "n": len(diffs)}


def coverage(rows, mode):
    n = len(rows)
    abst = sum(1 for r in rows if r["modes"][mode]["score"]["abstained"])
    return 1 - abst / n if n else 0.0


def cit_precision(rows, mode):
    vals = [r["modes"][mode]["score"]["citation_precision"] for r in rows
            if r["modes"][mode]["score"]["citation_precision"] is not None]
    return float(np.mean(vals)) if vals else None


def trust_tax(rows, mode):
    lat, in_tok, out_tok, calls = [], [], [], []
    for r in rows:
        res = r["modes"][mode]["result"]
        lat.append(res["latency_s"])
        in_tok.append(res["usage"]["input_tokens"])
        out_tok.append(res["usage"]["output_tokens"])
        calls.append(res["usage"]["n_calls"])
    return {"mean_latency_s": float(np.mean(lat)),
            "mean_input_tokens": float(np.mean(in_tok)),
            "mean_output_tokens": float(np.mean(out_tok)),
            "mean_calls": float(np.mean(calls)),
            "total_tokens": int(np.sum(in_tok) + np.sum(out_tok))}


def hard_case_breakdown(rows, mode):
    out = {}
    for r in rows:
        if not r.get("hard_case"):
            continue
        s = r["modes"][mode]["score"]
        out[r["qid"]] = {"csr": s["csr"], "abstained": s["abstained"],
                         "n_citations": s["n_citations"]}
    return out


def report(results_path: str):
    rows = json.load(open(results_path))
    modes = list(rows[0]["modes"].keys())

    print("=" * 72)
    print("AGGREGATE RESULTS")
    print("=" * 72)
    print(f"{'mode':<10}{'CSR':>8}  {'95% CI':<18}{'cov':>6}{'cit_prec':>10}{'lat_s':>8}{'tokens':>10}")
    print("-" * 72)
    for m in modes:
        b = bootstrap_csr(rows, m)
        cov = coverage(rows, m)
        cp = cit_precision(rows, m)
        tt = trust_tax(rows, m)
        if b:
            print(f"{m:<10}{b['mean']:>8.3f}  [{b['ci_lo']:.3f},{b['ci_hi']:.3f}] "
                  f"{cov:>6.2f}{(cp or 0):>10.3f}{tt['mean_latency_s']:>8.1f}"
                  f"{tt['mean_input_tokens']+tt['mean_output_tokens']:>10.0f}")
        else:
            print(f"{m:<10}     n/a")

    print("\n" + "=" * 72)
    print("PAIRED COMPARISONS vs baseline (CSR delta, 95% CI, one-sided p)")
    print("=" * 72)
    for m in modes:
        if m == "baseline":
            continue
        d = paired_diff(rows, "baseline", m)
        if d:
            print(f"{m:<10}  delta={d['mean_diff']:+.3f}  "
                  f"[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]  "
                  f"p={d['p_one_sided']:.3f}  n={d['n']}")

    print("\n" + "=" * 72)
    print("HARD CASES (per-question CSR by mode)")
    print("=" * 72)
    hards = sorted({r["qid"] for r in rows if r.get("hard_case")})
    print(f"{'qid':<28} " + "  ".join(f"{m:>10}" for m in modes))
    for h in hards:
        line = f"{h:<28} "
        for m in modes:
            row = next(r for r in rows if r["qid"] == h)
            s = row["modes"][m]["score"]
            cell = "ABST" if s["abstained"] else (
                f"{s['csr']:.2f}" if s["csr"] is not None else "n/a")
            line += f"  {cell:>10}"
        print(line)

    # save summary
    summary = {
        "modes": {m: {"csr": bootstrap_csr(rows, m),
                      "coverage": coverage(rows, m),
                      "citation_precision": cit_precision(rows, m),
                      "trust_tax": trust_tax(rows, m),
                      "hard_cases": hard_case_breakdown(rows, m)}
                  for m in modes},
        "paired_vs_baseline": {m: paired_diff(rows, "baseline", m)
                               for m in modes if m != "baseline"},
    }
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote summary.json")

    # Pareto plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in modes:
            b = bootstrap_csr(rows, m)
            tt = trust_tax(rows, m)
            if b is None:
                continue
            ax.errorbar(tt["mean_latency_s"], b["mean"],
                        yerr=[[b["mean"] - b["ci_lo"]], [b["ci_hi"] - b["mean"]]],
                        fmt="o", capsize=4, label=m)
            ax.annotate(m, (tt["mean_latency_s"], b["mean"]),
                        xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Mean latency per question (s)")
        ax.set_ylabel("Citation Support Rate")
        ax.set_title("Faithfulness vs Trust Tax")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("pareto.png", dpi=150)
        print("Wrote pareto.png")
    except Exception as e:
        print(f"(skipping plot: {e})")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "results.json")
