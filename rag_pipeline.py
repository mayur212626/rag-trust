"""
rag_pipeline.py

RAG baseline plus three interventions:
  baseline: vanilla RAG with inline citation prompt
  cove:     Chain-of-Verification adapted to citation grounding
            (Dhuliawala et al. 2024)
  nli:      DeBERTa-v3 NLI verifier; drop claims with entailment prob below 0.5
  quote:    Quote-then-cite re-prompt (one extra LLM call)

All four modes share the same retriever, prompt, and parsing so that the
only thing that varies is the verification step.

Dependencies:
    pip install anthropic rank_bm25 transformers torch sentencepiece
"""

import os
import re
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi

# LLM client
from anthropic import Anthropic
_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

GEN_MODEL = os.environ.get("GEN_MODEL", "claude-haiku-4-5-20251001")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")

# NLI verifier (lazy-loaded)
_nli_pipeline = None
def _get_nli():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "text-classification",
            model="microsoft/deberta-v3-large-mnli",
            top_k=None,
        )
    return _nli_pipeline


# Data classes

@dataclass
class Chunk:
    doc_id: str
    text: str

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    n_calls: int = 0
    def add(self, other: "TokenUsage"):
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.n_calls += other.n_calls

@dataclass
class RAGResult:
    question: str
    retrieved: List[Chunk]
    answer: str
    citations: List[str]
    abstained: bool = False
    latency_s: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    intermediate: Dict = field(default_factory=dict)


# Retrieval

def _tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())

class BM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(scores, self.chunks), key=lambda x: -x[0])
        return [c for _, c in ranked[:k]]


# Prompts

BASELINE_SYS = (
    "You are a careful research assistant. Answer the user's question using ONLY the provided sources. "
    "After every factual claim, append a citation in the form [doc_id] referring to the source you used. "
    "If multiple sources support a claim, cite them all: [doc_a][doc_b]. Do not cite a source you did not "
    "actually rely on. If the sources do not answer the question, say "
    "\"I cannot answer from these sources.\""
)

def _format_context(chunks: List[Chunk]) -> str:
    return "\n\n".join(f"[{c.doc_id}]\n{c.text}" for c in chunks)


# LLM call wrapper with usage tracking

def _call_llm(system: str, user: str, model: str = None, max_tokens: int = 800) -> Tuple[str, TokenUsage]:
    model = model or GEN_MODEL
    resp = _client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text
    usage = TokenUsage(
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        n_calls=1,
    )
    return text, usage


# Citation extraction and claim segmentation

CITE_RE = re.compile(r"\[([a-zA-Z0-9_\-\.]+)\]")
SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")

def extract_citations(answer: str) -> List[str]:
    return CITE_RE.findall(answer)

def segment_claims(answer: str) -> List[Dict]:
    """Sentence-level segmentation, keeping citations attached to each sentence."""
    out = []
    for sent in SENT_SPLIT.split(answer.strip()):
        cits = CITE_RE.findall(sent)
        clean = CITE_RE.sub("", sent).strip()
        if cits and clean:
            out.append({"claim": clean, "cited_ids": list(dict.fromkeys(cits))})
    return out


# Pipeline

class RAGPipeline:
    def __init__(self, chunks: List[Chunk], mode: str = "baseline", k: int = 5):
        assert mode in ("baseline", "cove", "nli", "quote")
        self.retriever = BM25Retriever(chunks)
        self.mode = mode
        self.k = k

    # Baseline
    def _baseline(self, q: str, ctx: List[Chunk]) -> RAGResult:
        t0 = time.time()
        user = f"Sources:\n{_format_context(ctx)}\n\nQuestion: {q}\n\nAnswer with inline citations."
        ans, usage = _call_llm(BASELINE_SYS, user)
        return RAGResult(
            question=q, retrieved=ctx, answer=ans,
            citations=extract_citations(ans),
            latency_s=time.time() - t0, usage=usage,
        )

    # CoVe
    def _cove(self, q: str, ctx: List[Chunk]) -> RAGResult:
        t0 = time.time()
        draft = self._baseline(q, ctx)
        usage = TokenUsage(); usage.add(draft.usage)

        # decompose
        decomp_prompt = (
            "Below is a draft answer with inline citations. Extract every distinct claim and the "
            "doc_ids cited for it. Output a JSON list of objects with keys 'claim' and 'cited_ids'.\n\n"
            f"Draft:\n{draft.answer}\n\nJSON only."
        )
        decomp_raw, u = _call_llm("You output only valid JSON.", decomp_prompt, max_tokens=600)
        usage.add(u)
        try:
            m = re.search(r"\[.*\]", decomp_raw, re.S)
            claims = json.loads(m.group(0)) if m else []
        except Exception:
            claims = []

        # verify
        chunk_lookup = {c.doc_id: c.text for c in ctx}
        verifications, verified = [], []
        for item in claims:
            cl = item.get("claim", "")
            cited = item.get("cited_ids", [])
            evidence = "\n".join(f"[{cid}] {chunk_lookup.get(cid, '<MISSING>')}" for cid in cited)
            v_prompt = (
                f"Claim: {cl}\nCited evidence:\n{evidence}\n\n"
                "Does the evidence LITERALLY support the claim? Reply on the FIRST line with one of: "
                "SUPPORTED or NOT_SUPPORTED, then a verbatim quote (if SUPPORTED) or one-sentence reason (if not)."
            )
            v, u = _call_llm("You verify factual claims strictly against evidence.", v_prompt, max_tokens=200)
            usage.add(u)
            first = v.strip().splitlines()[0].upper() if v.strip() else ""
            verifications.append({"claim": cl, "cited": cited, "raw": v})
            if first.startswith("SUPPORTED"):
                verified.append((cl, cited))

        if not verified:
            return RAGResult(
                question=q, retrieved=ctx,
                answer="I cannot answer reliably from these sources.",
                citations=[], abstained=True,
                latency_s=time.time() - t0, usage=usage,
                intermediate={"draft": draft.answer, "verifications": verifications},
            )

        # revise
        revise = "Rewrite the answer using ONLY these verified claims with their citations.\n\n"
        revise += "\n".join(f"- {c} {''.join(f'[{i}]' for i in ids)}" for c, ids in verified)
        revise += f"\n\nQuestion: {q}"
        final, u = _call_llm(BASELINE_SYS, revise, max_tokens=400)
        usage.add(u)
        return RAGResult(
            question=q, retrieved=ctx, answer=final,
            citations=extract_citations(final),
            latency_s=time.time() - t0, usage=usage,
            intermediate={"draft": draft.answer, "verifications": verifications},
        )

    # NLI verifier
    def _nli(self, q: str, ctx: List[Chunk]) -> RAGResult:
        t0 = time.time()
        draft = self._baseline(q, ctx)
        usage = TokenUsage(); usage.add(draft.usage)

        nli = _get_nli()
        chunk_lookup = {c.doc_id: c.text for c in ctx}
        kept_sents, dropped = [], []
        nli_log = []
        for sent in SENT_SPLIT.split(draft.answer.strip()):
            cits = CITE_RE.findall(sent)
            if not cits:
                continue
            premise = " ".join(chunk_lookup.get(cid, "") for cid in cits)
            hypothesis = CITE_RE.sub("", sent).strip()
            if not premise or not hypothesis:
                continue
            scores = nli(f"{premise} [SEP] {hypothesis}")
            score_dict = {s["label"].upper(): s["score"] for s in scores[0]}
            ent = score_dict.get("ENTAILMENT", 0.0)
            nli_log.append({"sent": sent, "ent": ent, "scores": score_dict})
            if ent >= 0.5:
                kept_sents.append(sent)
            else:
                dropped.append(sent)

        if not kept_sents:
            return RAGResult(
                question=q, retrieved=ctx,
                answer="I cannot answer reliably from these sources.",
                citations=[], abstained=True,
                latency_s=time.time() - t0, usage=usage,
                intermediate={"draft": draft.answer, "nli": nli_log},
            )
        final = " ".join(kept_sents)
        return RAGResult(
            question=q, retrieved=ctx, answer=final,
            citations=extract_citations(final),
            latency_s=time.time() - t0, usage=usage,
            intermediate={"draft": draft.answer, "nli": nli_log, "dropped": dropped},
        )

    # Quote-then-cite
    def _quote(self, q: str, ctx: List[Chunk]) -> RAGResult:
        t0 = time.time()
        draft = self._baseline(q, ctx)
        usage = TokenUsage(); usage.add(draft.usage)

        sources_block = _format_context(ctx)
        prompt = (
            f"Sources:\n{sources_block}\n\n"
            f"Draft answer:\n{draft.answer}\n\n"
            "For every claim in the draft, find a verbatim span of at most 25 words from the cited "
            "chunk that LITERALLY supports it. If you cannot find such a span, drop the claim. "
            "Re-output the answer keeping only verified claims with citations. "
            "If nothing remains, output exactly: I cannot answer from these sources."
        )
        final, u = _call_llm(BASELINE_SYS, prompt, max_tokens=600)
        usage.add(u)
        abstained = "cannot answer" in final.lower()[:60]
        return RAGResult(
            question=q, retrieved=ctx, answer=final,
            citations=[] if abstained else extract_citations(final),
            abstained=abstained,
            latency_s=time.time() - t0, usage=usage,
            intermediate={"draft": draft.answer},
        )

    # Dispatch
    def answer(self, question: str) -> RAGResult:
        ctx = self.retriever.retrieve(question, k=self.k)
        if self.mode == "baseline":
            return self._baseline(question, ctx)
        if self.mode == "cove":
            return self._cove(question, ctx)
        if self.mode == "nli":
            return self._nli(question, ctx)
        if self.mode == "quote":
            return self._quote(question, ctx)
        raise ValueError(self.mode)


def serialize_result(r: RAGResult) -> Dict:
    d = asdict(r)
    d["retrieved"] = [asdict(c) for c in r.retrieved]
    return d
