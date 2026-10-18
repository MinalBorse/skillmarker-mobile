import os
import tempfile

# Isolated, offline and free: temp data folder, no LLM key, offline text model.
os.environ["SKILLMARKER_DATA_DIR"] = tempfile.mkdtemp(prefix="skillmarker_test_")
os.environ["SKILLMARKER_MOCK_LLM"] = "1"
os.environ["EMBED_BACKEND"] = "hash"
os.environ.pop("OPENROUTER_API_KEY", None)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def sample_db():
    from skillmarker import config, extract, sample
    from skillmarker.embeddings import ensure_embeddings

    postings, truth = sample.generate(n=600, seed=7)
    sample.write(postings, truth)
    extract.run(config.SAMPLE_DB, method="rules", verbose=False)
    ensure_embeddings(config.SAMPLE_DB)
    return config.SAMPLE_DB
