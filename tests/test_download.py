"""Shard sibling matching helpers."""

from spark_llm.download import is_shard_sibling, shard_base


def test_shard_base() -> None:
    assert shard_base("gpt-oss-120b-mxfp4-00001-of-00003.gguf") == "gpt-oss-120b-mxfp4"
    assert shard_base("alone.gguf") == "alone"


def test_is_shard_sibling() -> None:
    primary = "gpt-oss-120b-mxfp4-00001-of-00003.gguf"
    assert is_shard_sibling(primary, primary)
    assert is_shard_sibling("gpt-oss-120b-mxfp4-00002-of-00003.gguf", primary)
    assert is_shard_sibling("gpt-oss-120b-mxfp4-00003-of-00003.gguf", primary)
    assert not is_shard_sibling("other-00001-of-00003.gguf", primary)
    assert not is_shard_sibling("readme.md", primary)
