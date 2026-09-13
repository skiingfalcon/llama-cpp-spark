"""Halo release-zip installer against a fake GitHub API (httpx.MockTransport; no network)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from spark_llm.config import Settings
from spark_llm.platforms.base import BuildOptions
from spark_llm.platforms.halo import install as inst
from spark_llm.platforms.halo.platform import load_build_info

TAG = "b10919"
COMMIT = "82d6bb284d1f"


def _zip(prefix: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{prefix}llama-server.exe", b"MZ fake")
        zf.writestr(f"{prefix}llama-bench.exe", b"MZ fake")
        zf.writestr(f"{prefix}ggml-vulkan.dll", b"dll")
    return buf.getvalue()


def _release(tag: str, names: list[str], repo: str = "ggml-org/llama.cpp") -> dict:
    return {
        "tag_name": tag,
        "published_at": "2026-09-12T02:18:40Z",
        "target_commitish": "d3146f2b56c2",
        "assets": [
            {
                "name": n,
                "size": 30 << 20,
                "browser_download_url": f"https://dl.example/{repo}/{tag}/{n}",
            }
            for n in names
        ],
    }


OFFICIAL_ASSETS = [
    f"llama-{TAG}-bin-win-cpu-x64.zip",
    f"llama-{TAG}-bin-win-vulkan-x64.zip",
    f"llama-{TAG}-bin-win-rocm-10.0-x64.zip",
    "cudart-llama-bin-win-cuda-13.3-x64.zip",
]
LEMONADE_ASSETS = [
    "llama-b1328-windows-rocm-gfx1150-x64.zip",
    "llama-b1328-windows-rocm-gfx1151-x64.zip",
    "llama-b1328-ubuntu-rocm-gfx1151-x64.zip",
]


def _client(zip_prefix: str = "", releases_pages: list[list[dict]] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://dl.example/"):
            return httpx.Response(200, content=_zip(zip_prefix))
        if url.endswith(f"/repos/ggml-org/llama.cpp/releases/tags/{TAG}"):
            return httpx.Response(200, json=_release(TAG, OFFICIAL_ASSETS))
        if url.endswith("/repos/lemonade-sdk/llamacpp-rocm/releases/latest"):
            return httpx.Response(
                200, json=_release("b1328", LEMONADE_ASSETS, repo="lemonade-sdk/llamacpp-rocm")
            )
        if "/repos/ggml-org/llama.cpp/commits/" in url:
            return httpx.Response(
                200, json={"commit": {"committer": {"date": "2026-09-11T22:53:07Z"}}}
            )
        if "/repos/ggml-org/llama.cpp/releases?" in url:
            page = int(request.url.params.get("page", "1"))
            pages = releases_pages or [[]]
            body = pages[page - 1] if page <= len(pages) else []
            return httpx.Response(200, json=body)
        return httpx.Response(404, json={"message": "not found", "url": url})

    return httpx.Client(transport=httpx.MockTransport(handler))


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        vendor_dir=tmp_path / "vendor", models_dir=tmp_path / "m", state_dir=tmp_path / "s"
    )


def test_pick_asset_patterns_and_override() -> None:
    assets = _release(TAG, OFFICIAL_ASSETS)["assets"]
    assert inst.pick_asset(assets, inst.ASSET_PATTERNS[("official", "vulkan")], None)[
        "name"
    ].endswith("win-vulkan-x64.zip")
    assert inst.pick_asset(assets, inst.ASSET_PATTERNS[("official", "rocm")], None)[
        "name"
    ].endswith("win-rocm-10.0-x64.zip")
    lem = _release("b1328", LEMONADE_ASSETS)["assets"]
    assert inst.pick_asset(lem, inst.ASSET_PATTERNS[("lemonade", "rocm")], None)["name"].endswith(
        "gfx1151-x64.zip"
    )
    assert (
        inst.pick_asset(assets, inst.ASSET_PATTERNS[("official", "rocm")], OFFICIAL_ASSETS[0])[
            "name"
        ]
        == OFFICIAL_ASSETS[0]
    )
    with pytest.raises(RuntimeError, match="no asset matches"):
        inst.pick_asset(lem, inst.ASSET_PATTERNS[("official", "vulkan")], None)
    with pytest.raises(RuntimeError, match="not in release"):
        inst.pick_asset(assets, inst.ASSET_PATTERNS[("official", "vulkan")], "nope.zip")


def test_resolve_tag_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    assert inst.resolve_tag(client, COMMIT, "b12345").tag == "b12345"
    monkeypatch.setattr(inst, "pinned_release", lambda: "b10919")
    r = inst.resolve_tag(client, COMMIT, None)
    assert r.tag == "b10919" and r.note and "LLAMA_CPP_RELEASE" in r.note


def test_resolve_tag_walks_releases_for_nearest_later(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inst, "pinned_release", lambda: None)
    rels = [
        {"tag_name": "b10920", "published_at": "2026-09-12T03:00:00Z", "target_commitish": "aaaa"},
        {
            "tag_name": "b10919",
            "published_at": "2026-09-12T02:18:40Z",
            "target_commitish": "d3146f2b56c2",
        },
        {
            "tag_name": "b10917",
            "published_at": "2026-09-11T20:30:32Z",
            "target_commitish": "8ea290247c87",
        },
    ]
    r = inst.resolve_tag(_client(releases_pages=[rels]), COMMIT, None)
    assert r.tag == "b10919" and "nearest later" in (r.note or "")
    exact = [
        {
            "tag_name": "b10918",
            "published_at": "2026-09-11T23:00:00Z",
            "target_commitish": COMMIT + "abc",
        }
    ]
    r = inst.resolve_tag(_client(releases_pages=[exact]), COMMIT, None)
    assert r.tag == "b10918" and r.note is None


@pytest.mark.parametrize("prefix", ["", "build/bin/"])
def test_install_both_backends_writes_build_json(
    tmp_path: Path, prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inst, "pinned_release", lambda: TAG)
    monkeypatch.setattr(inst, "pinned_commit", lambda: COMMIT)
    # The fake exe cannot run on Linux; probe() returns empty fields, which is the tolerated path.
    settings = _settings(tmp_path)
    rc = inst.install(settings, BuildOptions(), client=_client(zip_prefix=prefix))
    assert rc == 0
    info = load_build_info(settings)
    assert set(info["backends"]) == {"vulkan", "rocm"}
    vk = info["backends"]["vulkan"]
    assert vk["tag"] == TAG and vk["source"] == "official"
    assert (Path(vk["bin_dir"]) / "llama-server.exe").is_file()
    assert Path(vk["bin_dir"]).is_relative_to(tmp_path / "vendor" / f"{TAG}-vulkan")
    assert info["backends"]["rocm"]["asset"].endswith("win-rocm-10.0-x64.zip")


def test_install_lemonade_source_and_idempotency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inst, "pinned_release", lambda: TAG)
    settings = _settings(tmp_path)
    assert (
        inst.install(settings, BuildOptions(backends=["rocm"], source="lemonade"), client=_client())
        == 0
    )
    hip = load_build_info(settings)["backends"]["rocm"]
    assert hip["source"] == "lemonade" and hip["tag"] == "b1328" and "gfx1151" in hip["asset"]
    # second run without --force skips (bin dir still present)
    calls: list[str] = []
    orig = inst.download

    def spy(client, url, dest):  # type: ignore[no-untyped-def]
        calls.append(url)
        return orig(client, url, dest)

    monkeypatch.setattr(inst, "download", spy)
    assert (
        inst.install(settings, BuildOptions(backends=["rocm"], source="lemonade"), client=_client())
        == 0
    )
    assert calls == []
    assert (
        inst.install(
            settings,
            BuildOptions(backends=["rocm"], source="lemonade", force=True),
            client=_client(),
        )
        == 0
    )
    assert len(calls) == 1


def test_install_reinstalls_when_source_or_tag_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inst, "pinned_release", lambda: TAG)
    settings = _settings(tmp_path)
    assert inst.install(settings, BuildOptions(backends=["rocm"]), client=_client()) == 0
    assert load_build_info(settings)["backends"]["rocm"]["source"] == "official"
    # asking for the lemonade build must not be skipped because an official one exists
    assert (
        inst.install(settings, BuildOptions(backends=["rocm"], source="lemonade"), client=_client())
        == 0
    )
    assert load_build_info(settings)["backends"]["rocm"]["source"] == "lemonade"


def test_resolve_tag_falls_back_to_latest_when_pin_is_newer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inst, "pinned_release", lambda: None)
    older = [
        {"tag_name": "b10917", "published_at": "2026-09-11T20:30:32Z", "target_commitish": "8ea2"}
    ]
    r = inst.resolve_tag(_client(releases_pages=[older]), COMMIT, None)
    assert r.tag == "b10917" and "newer than any release" in (r.note or "")


def test_install_rejects_bad_options(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert inst.install(settings, BuildOptions(backends=["cuda"]), client=_client()) == 2
    assert inst.install(settings, BuildOptions(source="ftp"), client=_client()) == 2


def test_pinned_release_parses_comment_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "LLAMA_CPP_RELEASE"
    f.write_text("# comment\n\nb10919  # nearest\n")
    monkeypatch.setattr(inst, "repo_root", lambda: tmp_path)
    assert inst.pinned_release() == "b10919"
    f.unlink()
    assert inst.pinned_release() is None
    assert json.loads(json.dumps({"ok": True}))  # keep json import honest for the fixture helpers
