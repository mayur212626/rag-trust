# Pre-Registration: RAG Citation Faithfulness Experiment

**Mayur Patil | Committed May 3, 2026 before any API calls**

I'm writing this before running anything. The point is to lock my metric, my hypotheses, and the specific questions I think will be hard, so I can't change the rules after seeing the results. The commit hash of this file is my proof.

---

## The problem I'm testing

RAG systems are supposed to be "grounded" because they cite retrieved documents. My question is whether the citations are actually faithful — whether the cited document really supports the claim attached to it. I think the baseline will fail on this more than people expect, and I want to test whether two simple verifiers can fix it.

## The metric: Citation Support Rate (CSR)

For each answer the model produces, I split it into sentence-level cited claims. For every (claim, cited chunk) pair, a judge model scores it as one of:
- SUPPORTED: the chunk literally supports the claim
- PARTIAL: the chunk is related but the claim goes beyond what the chunk says
- UNSUPPORTED: the chunk doesn't back up the claim
- CONTRADICTED: the chunk says something opposite

CSR = number of SUPPORTED pairs / total cited pairs.

Only SUPPORTED counts. PARTIAL is not a pass.

I'm also tracking citation precision (did the cited doc_id actually appear in the retrieved set?) separately, because hallucinated doc_ids are a different failure mode.

If a question gets answered with no citations at all, it doesn't contribute to CSR but it counts against coverage.

**Why this maps to trust:** citations are what users rely on to decide whether to trust an answer. If the citations are unreliable, the model is manufacturing false confidence. For legal or medical use I'd set the bar at 95% or above. The whole point of this experiment is to measure how far below that bar the baseline is and whether verifiers can close the gap.

## System setup

- Retriever: BM25 over 600 Wikipedia paragraphs from HotpotQA distractor validation set
- Generator: claude-haiku-4-5
- Judge: claude-opus-4-7 (different model family to reduce self-preference)
- Top-5 chunks per question
- 50 eval questions from HotpotQA dev + 5 hard cases below
- 10 held-out questions that I won't look at until after the main eval is done

## The two interventions I'm testing

**Chain-of-Verification (CoVe):** after the baseline produces a draft, decompose it into atomic claims with their cited doc_ids, verify each claim against its cited chunk, drop the ones that fail. Based on Dhuliawala et al. 2024.

**Quote-then-cite:** one follow-up call that asks the model to find a verbatim span of 25 words or fewer in each cited chunk for each claim. If it can't find the span, drop the claim. My own prompt design, loosely inspired by Self-RAG (Asai et al. 2024).

## My hypotheses (locked before running)

H1: Both interventions improve CSR by at least 5 percentage points absolute.

H2: CoVe improves CSR more than quote-then-cite but at higher latency cost.

H3: Both interventions reduce coverage by at least 10 percentage points (some questions become abstentions).

I'm committing to reporting negative results. If CoVe doesn't improve CSR, that's the result, not a footnote.

## The 5 hard cases I predict the baseline will fail on

I'm predicting the baseline fails on at least 3 of these 5. "Fail" means CSR=None (no citations) or CSR=0.0.

**hard_01 (multi-hop numeric):** "What is the population difference between the largest cities of Australia and New Zealand?"
Predicted failure: the model will need to pull numbers from two different chunks and do arithmetic. I expect it to cite one chunk for both numbers, or cite neither.

**hard_02 (negation):** "Which of argon, nitrogen, neon, krypton is NOT a noble gas?"
Predicted failure: every chunk confirms that argon/neon/krypton ARE noble gases. None say "nitrogen is NOT a noble gas." The model will struggle to cite support for a negation.

**hard_03 (entity collision):** "What university does statistician Michael I. Jordan teach at?"
Predicted failure: my corpus has chunks about Michael Jordan the basketball player. I expect the model to either get confused or cite basketball-player chunks for statistician facts.

**hard_04 (temporal staleness):** "Who is the current CEO of Twitter?"
Predicted failure: the Wikipedia chunk I have is from 2019. The model will confidently cite a stale fact as current.

**hard_05 (aggregation):** "List every Turing Award winner from 2015 to 2018."
Predicted failure: no single chunk has all four years. The model will get partial lists from one chunk and fabricate the rest without citation.

## Stopping rules

- Sample size locked at 50 + 10 held-out. No expanding after seeing results.
- Judge prompt frozen after pilot validation. No changes mid-experiment.
- Hard cases reported regardless of aggregate result, even if every single one passes.
- Code and prompts committed to git before the first eval run.
