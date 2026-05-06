# RAG Citation Faithfulness, Trustworthy AI Final Project

A pre-registered, reproducible study of three verifier interventions against a baseline RAG pipeline, evaluated on multi-hop QA with a faithfulness vs cost analysis.

## Files

| File                  | Purpose |
|-----------------------|---------|
| `preregistration.md`  | **Commit FIRST.** Locks the metric, hypotheses, hard cases, and judge protocol before any code runs. |
| `references.bib`      | BibTeX for every cited source. Every entry was hand-verified against ACL Anthology, arXiv, MIT Press, or court records on May 3, 2026. |
| `rag_pipeline.py`     | Baseline RAG plus three interventions (CoVe, NLI, quote-then-cite). |
| `judge.py`            | LLM-as-judge for citation faithfulness. Includes a Cohen's kappa agreement utility. |
| `prep_data.py`        | Pulls HotpotQA dev, builds the corpus, the 50-question eval set with 5 hard cases, and the 10-question held-out set. |
| `run_eval.py`         | Runs all four modes over the eval set and dumps `results.json`. |
| `analysis.py`         | Bootstrap CIs, paired tests, hard-case breakdown, Pareto plot. |
| `report.md`           | AAAI/AIES-style report draft. Numbers are `[fill]` until the eval completes. |

## Verified citations

Every reference in `references.bib` and `report.md` was verified on May 3, 2026 against an authoritative source:

- Dhuliawala et al. (CoVe): ACL Anthology 2024.findings-acl.212, arXiv 2309.11495.
- Asai et al. (Self-RAG): ICLR 2024, arXiv 2310.11511.
- Min et al. (FActScore): EMNLP 2023 Anthology 2023.emnlp-main.741, arXiv 2305.14251.
- Gao et al. (ALCE): EMNLP 2023 Anthology 2023.emnlp-main.398, arXiv 2305.14627.
- Rashkin et al. (AIS): *Computational Linguistics* 49(4):777 to 840, MIT Press 2023, DOI 10.1162/coli_a_00486.
- Yang et al. (HotpotQA): EMNLP 2018 Anthology D18-1259, arXiv 1809.09600.
- Ji et al. (Hallucination Survey): *ACM Computing Surveys* 55(12), Article 248, DOI 10.1145/3571730.
- *Mata v. Avianca, Inc.*, 678 F. Supp. 3d 443 (S.D.N.Y. 2023). Court reporter, opinion dated June 22, 2023, Judge P. Kevin Castel.

## Quickstart

```bash
pip install anthropic rank_bm25 datasets transformers torch sentencepiece numpy matplotlib
export ANTHROPIC_API_KEY=sk-ant-...

# 0. Lock the pre-registration first.
git add preregistration.md references.bib && git commit -m "Pre-registration locked"

# 1. Build corpus and questions (about 30 sec).
python prep_data.py

# 2. (Recommended) Validate the judge before locking it.
#    Run a small pilot, hand-label 30 (claim, chunk) pairs into pilot_pairs.jsonl,
#    then:
python -c "from judge import run_judge_agreement; run_judge_agreement('pilot_pairs.jsonl')"
# Require kappa_binary >= 0.6 before continuing.

# 3. Full evaluation. Spends a few dollars on the API. Takes 1 to 2 hours.
python run_eval.py --corpus corpus.jsonl --questions questions.jsonl

# 4. Analysis.
python analysis.py results.json     # writes summary.json + pareto.png

# 5. Held-out qualitative pass. Run AFTER step 4 numbers are locked.
python run_eval.py --corpus corpus.jsonl --questions heldout.jsonl --out heldout_results.json
```

## One-day workflow

| Hour    | Task |
|---------|------|
| 0       | Commit `preregistration.md`. |
| 0 to 1  | `prep_data.py`. Eyeball hard cases. Replace any that are genuinely impossible vs. the corpus. |
| 1 to 2  | Pilot baseline run on 5 questions. Hand-label 30 (claim, chunk) pairs. Run judge agreement; iterate prompt only if kappa < 0.6, then **lock it**. |
| 2 to 7  | Full `run_eval.py`. |
| 7 to 9  | `analysis.py`. Fill in Tables 1 and 2 of `report.md`. |
| 9 to 10 | Held-out qualitative pass. Write three vignettes for section 5 of the report. |
| 10 to 12| Polish report, finalize trust-tax discussion, generate Pareto plot. |

## Things that will go wrong, calibrate now

- **Judge disagrees on PARTIAL.** The pre-registration treats only SUPPORTED as supported. Stick with it.
- **CoVe over-abstains.** Coverage is reported as a first-class metric so this is visible, not hidden.
- **Hallucinated `doc_id`s.** Citation precision catches these as a separate metric.
- **NLI verifier is over-strict on multi-sentence claims.** Acceptable. The pre-registration commits to this exact decision rule.
- **Hard cases need corpus support.** If `hard_04_temporal` (current CEO) genuinely lacks a stale doc in your retrieved corpus, the right baseline behavior is abstention. The failure mode is *confident citation of stale fact*, not absence of an answer.

## Out of scope (acknowledged in section 7 of the report)

- Faithfulness is not truth.
- English only.
- Single retriever (BM25). A dense retriever may shift failure modes.
- Single model family for the generator.
