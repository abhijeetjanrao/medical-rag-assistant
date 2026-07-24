import sys
import os
import chromadb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.ingest import chunk_pages, CHUNK_SIZE


def test_chunk_pages_preserves_page_number():
    pages = [{"text": "A" * 2000, "page_number": 3, "source": "doc.pdf"}]
    chunks = chunk_pages(pages)
    assert all(c["page_number"] == 3 for c in chunks)
    assert all(c["source"] == "doc.pdf" for c in chunks)


def test_chunk_pages_respects_size_roughly():
    pages = [{"text": "word " * 1000, "page_number": 1, "source": "doc.pdf"}]
    chunks = chunk_pages(pages)
    # allow slack since splitter breaks on separators, not exact char count
    assert all(len(c["text"]) <= CHUNK_SIZE * 1.5 for c in chunks)


def test_empty_page_produces_no_chunks():
    pages = [{"text": "", "page_number": 1, "source": "doc.pdf"}]
    chunks = chunk_pages(pages)
    assert chunks == []


def test_multiple_pages_produce_chunks_for_each():
    pages = [
        {"text": "Some content here. " * 50, "page_number": 1, "source": "doc.pdf"},
        {"text": "Different content here. " * 50, "page_number": 2, "source": "doc.pdf"},
    ]
    chunks = chunk_pages(pages)
    page_numbers = {c["page_number"] for c in chunks}
    assert page_numbers == {1, 2}
