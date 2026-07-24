"""
Hybrid retrieval: dense (vector) + sparse (BM25) search, combined with
Reciprocal Rank Fusion. Pure vector search misses exact terms — drug names,
dosages, ICD codes — that clinicians actually search for. BM25 catches those.
"""
import os
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "medical_docs"
TOP_K = int(os.getenv("TOP_K", 5))
RRF_K = 60  # standard RRF smoothing constant


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page_number: int
    score: float


class HybridRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=PERSIST_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="pritamdeka/S-PubMedBert-MS-MARCO"
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embed_fn
        )

        # Pull everything once to build the BM25 index in memory.
        # Fine up to a few hundred thousand chunks; move to a real search
        # engine (Elasticsearch/OpenSearch) if the corpus grows much beyond that.
        all_docs = self.collection.get(include=["documents", "metadatas"])
        self.doc_ids = all_docs["ids"]
        self.documents = all_docs["documents"]
        self.metadatas = all_docs["metadatas"]
        tokenized = [d.lower().split() for d in self.documents]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def _vector_search(self, query: str, k: int) -> list[tuple[str, float]]:
        results = self.collection.query(query_texts=[query], n_results=k)
        ids = results["ids"][0]
        distances = results["distances"][0]
        # Chroma returns cosine distance; convert to a similarity-style score
        return [(doc_id, 1 - dist) for doc_id, dist in zip(ids, distances)]

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def retrieve(self, query: str, k: int = TOP_K) -> list[RetrievedChunk]:
        vector_hits = self._vector_search(query, k=k * 2)
        bm25_hits = self._bm25_search(query, k=k * 2)

        # Reciprocal Rank Fusion — combine two ranked lists without needing
        # to normalize wildly different score scales (cosine sim vs BM25).
        fused_scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(vector_hits):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)
        for rank, (doc_id, _) in enumerate(bm25_hits):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (RRF_K + rank + 1)

        top_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        chunks = []
        for doc_id, score in top_ids:
            idx = id_to_idx[doc_id]
            meta = self.metadatas[idx]
            chunks.append(RetrievedChunk(
                text=self.documents[idx],
                source=meta["source"],
                page_number=meta["page_number"],
                score=round(score, 4),
            ))
        return chunks
