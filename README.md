# RAG Citation Faithfulness

**Mayur Patil | Trustworthy AI Final Project | May 2026**

This is the code for my experiment testing whether simple post-hoc verifiers can improve citation faithfulness in RAG systems. Short answer: one of them worked, one mostly didn't.

The full writeup is in `final_report.pdf`. This README just covers how to run the code.

## What's in here

- `prep_data.py` — pulls HotpotQA from HuggingFace and builds the corpus, eval questions, and held-out set
- `rag_pipeline.py` — the baseline RAG pipeline plus two interventions (CoVe and quote-then-cite)
- `judge.py` — LLM-as-judge that scores each (claim, cited chunk) pair as SUPPORTED/PARTIAL/UNSUPPORTED/CONTRADICTED
- `run_eval.py` — runs all three modes over the question set and saves everything to results.json
- `analysis.py` — computes bootstrap CIs, paired tests, and generates the Pareto plot
- `preregistration.md` — my metric definition, hypotheses, and hard cases, committed before I ran anything
- `references.bib` — full BibTeX for every paper cited

## How to run it

```bash
pip install anthropic rank_bm25 datasets numpy matplotlib
export ANTHROPIC_API_KEY=your-key-here

python prep_data.py
python run_eval.py --corpus corpus.jsonl --questions questions.jsonl --out results.json
python analysis.py results.json
```

The full eval takes about 1-2 hours and costs roughly $3-5 in API calls. It saves incrementally so if it crashes you can pick up where you left off.

For the held-out qualitative set:
```bash
python run_eval.py --corpus corpus.jsonl --questions heldout.jsonl --out heldout_results.json
```

## A few things that bit me

**CoVe's JSON parsing breaks on about 15% of questions.** When the decomposition step returns malformed JSON the pipeline falls back to the unverified baseline. I didn't log every failure which made it hard to separate "the approach is bad" from "my implementation was brittle." I'd fix this first in a next version.

**The judge is single-chunk strict.** A claim that requires reasoning across two chunks will always score PARTIAL or lower even when the reasoning is correct. This makes CSR numbers on multi-hop questions lower than they probably should be.

**Hard cases all failed the same way.** I expected the baseline to fabricate citations on the hard questions. Instead it answered without citing anything. The interventions abstained on most of them. Both behaviors are useless to the user but in different ways.

## Pre-registration

The `preregistration.md` file was committed at `175168f` before I ran any API calls. It locks the metric definition, both hypotheses, and the five hard cases with predicted failure modes. I did this so I couldn't change the rules after seeing the results.
