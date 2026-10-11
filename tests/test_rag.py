from nexusiq.rag.loaders import chunk_document, load_document, parse_front_matter
from nexusiq.rag.retriever import BM25Index, reciprocal_rank_fusion


def test_front_matter_parsing():
    meta, body = parse_front_matter("---\ntitle: Demo\naccess: hr_only\n---\n# Heading\ntext")
    assert meta == {"title": "Demo", "access": "hr_only"} and body.startswith("# Heading")


def test_unknown_access_value_falls_back_to_public():
    _, meta = load_document("x.md", b"---\naccess: top_secret\n---\nhello")
    assert meta["access"] == "public"


def test_chunks_keep_section_headings():
    text = "# Policy\nintro\n## Leave\n" + "Annual leave is 24 days. " * 5 + "\n## Travel\nHotel cap applies."
    _, meta = load_document("policy.md", text.encode())
    chunks = chunk_document(text, meta)
    sections = {c.metadata["section"] for c in chunks}
    assert any("Leave" in s for s in sections) and any("Travel" in s for s in sections)
    assert all(c.metadata["source"] == "policy.md" for c in chunks)


def test_bm25_prefers_exact_keyword_match():
    docs = ["notice period is 60 days", "annual leave is 24 days", "hotel limit is 8000"]
    scores = BM25Index(docs).scores("notice period")
    assert scores.index(max(scores)) == 0


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused["a"] == fused["b"] > fused["c"]
    assert fused["c"] > 0 and fused["d"] > 0


def test_ingestion_is_idempotent_and_searchable(engine):
    before = engine.kb.count()
    engine.kb.ingest_directory(engine.settings.kb_dir)
    assert engine.kb.count() == before > 0
    hits = engine.kb.search("how many days of annual leave", role="employee")
    assert hits and hits[0].metadata["source"] == "hr_leave_policy.md"


def test_access_control_is_applied_before_ranking(engine):
    employee = engine.kb.search("salary bands and bonus percentage", role="employee", k=8)
    assert all(h.metadata["access"] == "public" for h in employee)
    hr = engine.kb.search("salary bands and bonus percentage", role="hr_admin", k=8)
    assert hr[0].metadata["source"] == "compensation_bands.md"


def test_uploading_a_new_document_makes_it_searchable(engine):
    n = engine.kb.ingest_bytes("canteen.md", b"# Canteen\nThe cafeteria serves free filter coffee every morning at 9 am.")
    assert n >= 1
    hits = engine.kb.search("free filter coffee cafeteria", role="employee", k=2)
    assert hits[0].metadata["source"] == "canteen.md"
    engine.kb.delete_source("canteen.md")
    assert all(h.metadata["source"] != "canteen.md" for h in engine.kb.search("free filter coffee", role="employee"))
