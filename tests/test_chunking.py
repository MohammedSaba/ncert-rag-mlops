from app.chunking import split_sentences, chunk_pages_sentence


def test_abbreviations_do_not_split():
    assert len(split_sentences("Use e.g. a magnet. Then stop.")) == 2


def test_chunks_end_on_sentence_boundaries():
    text = " ".join(f"This is sentence number {i} about chemistry." for i in range(60))
    chunks = chunk_pages_sentence([{"page": 1, "text": text}], 300, 1)
    assert len(chunks) > 1
    assert all(c["text"].endswith(".") for c in chunks)
    assert all(len(c["text"]) <= 400 for c in chunks)