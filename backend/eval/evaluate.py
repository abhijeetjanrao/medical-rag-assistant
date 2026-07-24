"""
Evaluation harness. Run this against a labeled eval set to get the numbers
that go on your resume: retrieval precision@k, answer faithfulness, latency.

Eval set format (eval/eval_set.json):
[
  {
    "question": "What is the recommended isolation period for measles?",
    "ground_truth": "Isolate for 4 days after rash onset...",
    "expected_sources": ["cdc_measles_guidelines.pdf"]
  },
  ...
]

Run: python -m eval.evaluate
"""
import asyncio
import json
import os
import statistics
import time
import typing as t
from dataclasses import dataclass, field

import requests
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.run_config import RunConfig
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")

# -----------------------------------------------------------------------
# Custom Ragas LLM/embeddings wrappers that call the SDKs directly instead
# of going through langchain_google_genai / langchain_huggingface. Those
# packages have repeatedly broken across version combinations (missing
# kwargs support, missing transitive deps); calling google-generativeai
# and sentence-transformers directly here mirrors what main.py and
# rag/ingest.py already do successfully, sidestepping the issue entirely.
# -----------------------------------------------------------------------

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


@dataclass
class GeminiRagasLLM(BaseRagasLLM):
    model_name: str = "gemini-2.0-flash"
    run_config: RunConfig = field(default_factory=RunConfig, repr=False)
    multiple_completion_supported: bool = field(default=False, repr=False)

    def _call_gemini(self, prompt_text: str) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt_text)
        return response.text

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        text = self._call_gemini(prompt.to_string())
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._call_gemini, prompt.to_string())
        return LLMResult(generations=[[Generation(text=text)]])


class SentenceTransformerRagasEmbeddings(BaseRagasEmbeddings):
    """Free, local embeddings for Ragas's answer_relevancy metric — reuses
    the same biomedical model already used for retrieval in rag/ingest.py."""

    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        self.model = SentenceTransformer(model_name)
        self.run_config = RunConfig()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


_judge_llm = GeminiRagasLLM()
_judge_embeddings = SentenceTransformerRagasEmbeddings()


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH) as f:
        return json.load(f)


def run_queries(eval_set: list[dict]) -> tuple[list[dict], list[float]]:
    """Hits the live /chat endpoint for every eval question and records latency."""
    results = []
    latencies = []
    for item in eval_set:
        start = time.time()
        resp = requests.post(f"{API_URL}/chat", json={"query": item["question"]})
        resp.raise_for_status()
        data = resp.json()
        latencies.append(time.time() - start)

        results.append({
            "question": item["question"],
            "answer": data["answer"],
            "contexts": [s["excerpt"] for s in data["sources"]],
            "ground_truth": item["ground_truth"],
            "retrieved_sources": {s["document_name"] for s in data["sources"]},
            "expected_sources": set(item.get("expected_sources", [])),
        })
    return results, latencies


def retrieval_precision_at_k(results: list[dict]) -> float:
    """Fraction of queries where at least one expected source was retrieved."""
    hits = sum(
        1 for r in results
        if r["expected_sources"] & r["retrieved_sources"]
    )
    return hits / len(results) if results else 0.0


def run_ragas(results: list[dict]) -> dict:
    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        } for r in results
    ])
    scored = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        llm=_judge_llm,
        embeddings=_judge_embeddings,
    )
    return scored.to_pandas().mean(numeric_only=True).to_dict()


def main():
    eval_set = load_eval_set()
    print(f"Running {len(eval_set)} eval queries against {API_URL} ...")
    results, latencies = run_queries(eval_set)

    retrieval_p_at_k = retrieval_precision_at_k(results)
    ragas_scores = run_ragas(results)

    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)

    print("\n=== Evaluation results ===")
    print(f"Queries evaluated:        {len(eval_set)}")
    print(f"Retrieval precision@k:    {retrieval_p_at_k:.2%}")
    print(f"Faithfulness (RAGAS):     {ragas_scores.get('faithfulness', float('nan')):.2f}")
    print(f"Context precision (RAGAS):{ragas_scores.get('context_precision', float('nan')):.2f}")
    print(f"Answer relevancy (RAGAS): {ragas_scores.get('answer_relevancy', float('nan')):.2f}")
    print(f"Latency p50:              {p50*1000:.0f} ms")
    print(f"Latency p95:              {p95*1000:.0f} ms")

    with open("eval_results.json", "w") as f:
        json.dump({
            "retrieval_precision_at_k": retrieval_p_at_k,
            "ragas": ragas_scores,
            "latency_p50_ms": p50 * 1000,
            "latency_p95_ms": p95 * 1000,
            "n_queries": len(eval_set),
        }, f, indent=2)
    print("\nSaved to eval_results.json — these are your resume numbers.")


if __name__ == "__main__":
    main()
