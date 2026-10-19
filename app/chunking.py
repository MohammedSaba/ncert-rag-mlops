import re
from pypdf import PdfReader

# Periods that do NOT end a sentence (protected before splitting)
ABBREV = ["e.g.", "i.e.", "Fig.", "Figs.", "etc.", "vs.", "Dr.", "No.", "approx."]


def load_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"-\n(\w)", r"\1", text)   # fix hyphenated line breaks
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 50:
            pages.append({"page": i + 1, "text": text})
    return pages


def split_sentences(text):
    for a in ABBREV:
        text = text.replace(a, a.replace(".", "<DOT>"))
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“(\[])', text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def chunk_pages_sentence(pages, chunk_size=800, overlap_sents=1):
    """Pack whole sentences up to chunk_size chars; repeat the last
    `overlap_sents` sentences at the start of the next chunk."""
    chunks = []

    def emit(sents, page):
        text = " ".join(sents).strip()
        if len(text) > 50:
            chunks.append({"id": len(chunks), "page": page, "text": text})

    for p in pages:
        current, cur_len = [], 0
        for s in split_sentences(p["text"]):
            if current and cur_len + len(s) + 1 > chunk_size:
                emit(current, p["page"])
                tail = current[-overlap_sents:] if overlap_sents else []
                if sum(len(t) + 1 for t in tail) + len(s) > chunk_size:
                    tail = []                     # avoid duplicate/oversized chunks
                current, cur_len = list(tail), sum(len(t) + 1 for t in tail)
            current.append(s)
            cur_len += len(s) + 1
        if current:
            emit(current, p["page"])
    return chunks


def chunk_pages_fixed(pages, chunk_size=800, overlap=150):
    """Baseline: fixed-size character windows (kept for experiments)."""
    chunks = []
    for p in pages:
        text, start = p["text"], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append({"id": len(chunks), "page": p["page"], "text": chunk})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks