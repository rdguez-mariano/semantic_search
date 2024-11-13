import tempfile

from nqs.llm.rag_retriever import get_docs_from_scraped_data, scrape_new_data


def test_update_sources():
    tempdir = tempfile.mkdtemp(prefix="test_sources_")
    scrape_new_data(tempdir)
    docs = get_docs_from_scraped_data(tempdir)
    assert any(
        [
            isinstance(d.page_content, str) and d.page_content.strip() != ""
            for d in docs
        ]
    )
