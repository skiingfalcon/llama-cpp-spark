"""Install prebuilt llama.cpp Windows zips for the Halo box (``local-llm build`` on Windows).

Why zips and not a source build: the Spark compiles a pinned commit with CUDA because that is
the only way to get sm_121a kernels. On Windows the upstream project publishes a Vulkan zip and a
ROCm zip for every commit tag (``bNNNNN``), so there is nothing to compile and no toolchain to
install. Parity with the Spark pin is by tag: the exact tag when the pinned commit has one, else
the nearest later tag, and the delta is recorded in ``halo-build.json``.

Two sources for the HIP backend:
- ``official``: ``ggml-org/llama.cpp`` ``llama-<tag>-bin-win-rocm-<ver>-x64.zip`` (HIP runtime
  DLLs bundled). gfx1151 coverage is verified after extraction with ``--list-devices``.
- ``lemonade``: ``lemonade-sdk/llamacpp-rocm`` ``llama-<tag>-windows-rocm-gfx1151-x64.zip``, an
  AMD-backed build with a dedicated gfx1151 target and all ROCm libs; tracks its own upstream
  commit (recorded), so use it when the official zip does not see the GPU.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from spark_llm.config import Settings, repo_root
from spark_llm.console import err as console
from spark_llm.platforms.base import BuildOptions
from spark_llm.platforms.halo.platform import (
    BACKEND_ALIASES,
    BACKENDS,
    SERVER_EXE,
    HaloPlatform,
    load_build_info,
    parse_devices_output,
    parse_version_output,
    save_build_info,
)
from spark_llm.provenance import pinned_commit

API = "https://api.github.com"
OFFICIAL_REPO = "ggml-org/llama.cpp"
LEMONADE_REPO = "lemonade-sdk/llamacpp-rocm"
RELEASE_FILE = "LLAMA_CPP_RELEASE"
ASSET_PATTERNS = {
    ("official", "vulkan"): re.compile(r"^llama-.*-bin-win-vulkan-x64\.zip$"),
    ("official", "rocm"): re.compile(r"^llama-.*-bin-win-rocm-[\d.]+-x64\.zip$"),
    ("lemonade", "rocm"): re.compile(r"^llama-.*-windows-rocm-gfx1151-x64\.zip$"),
}
SOURCES = ("official", "lemonade")


@dataclass
class ResolvedTag:
    tag: str
    note: str | None  # None when the tag is exactly the pinned commit


def pinned_release() -> str | None:
    """First non-comment token of LLAMA_CPP_RELEASE, or None."""
    path = repo_root() / RELEASE_FILE
    if not path.is_file():
        return None
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            return s.split()[0]
    return None


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:  # unauthenticated API calls are limited to 60/hour; paging releases can hit that
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(client: httpx.Client, url: str, **params: Any) -> Any:
    r = client.get(url, params=params or None, headers=_headers())
    r.raise_for_status()
    try:
        return r.json()
    except ValueError as exc:  # HTML error page, proxy interstitial, ...
        raise RuntimeError(f"unexpected non-JSON response from {url}") from exc


def resolve_tag(client: httpx.Client, commit: str | None, explicit: str | None) -> ResolvedTag:
    """Explicit tag > LLAMA_CPP_RELEASE > exact tag of the pinned commit > nearest later tag."""
    if explicit:
        return ResolvedTag(explicit, None)
    release = pinned_release()
    if release:
        return ResolvedTag(release, None if not commit else f"from {RELEASE_FILE}")
    if not commit:
        latest = _get(client, f"{API}/repos/{OFFICIAL_REPO}/releases/latest")
        return ResolvedTag(latest["tag_name"], "no pin; latest release")
    info = _get(client, f"{API}/repos/{OFFICIAL_REPO}/commits/{commit}")
    commit_date = info["commit"]["committer"]["date"]
    nearest_later: dict[str, Any] | None = None
    latest_seen: dict[str, Any] | None = None
    for page in range(1, 11):
        rels = _get(client, f"{API}/repos/{OFFICIAL_REPO}/releases", per_page=100, page=page)
        if not rels:
            break
        latest_seen = latest_seen or rels[0]
        for rel in rels:
            if str(rel.get("target_commitish", "")).startswith(commit):
                return ResolvedTag(rel["tag_name"], None)
            if rel["published_at"] > commit_date:
                nearest_later = rel  # list is newest-first; keep overwriting toward the pin
            elif nearest_later is not None:
                return ResolvedTag(
                    nearest_later["tag_name"],
                    f"pinned commit {commit} has no tag; nearest later tag",
                )
            else:
                break
    if nearest_later is not None:
        return ResolvedTag(nearest_later["tag_name"], f"nearest later tag to {commit}")
    if latest_seen is not None:
        return ResolvedTag(
            latest_seen["tag_name"], f"pinned commit {commit} is newer than any release; latest"
        )
    raise RuntimeError(f"could not resolve a llama.cpp release tag for commit {commit}")


def pick_asset(
    assets: list[dict[str, Any]], pattern: re.Pattern[str], override: str | None
) -> dict[str, Any]:
    if override:
        for a in assets:
            if a["name"] == override:
                return a
        raise RuntimeError(f"asset {override!r} not in release")
    hits = [a for a in assets if pattern.match(a["name"])]
    if not hits:
        names = ", ".join(a["name"] for a in assets if "win" in a["name"]) or "none"
        raise RuntimeError(f"no asset matches {pattern.pattern}; windows assets: {names}")
    return hits[0]


def download(client: httpx.Client, url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url, follow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def extract(zip_path: Path, dest: Path) -> Path:
    """Unzip and return the directory containing llama-server.exe (zips differ in prefix)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    hits = sorted(dest.rglob(SERVER_EXE), key=lambda p: len(p.parts))
    if not hits:
        raise RuntimeError(f"{SERVER_EXE} not found in {zip_path.name}")
    return hits[0].parent


def probe(
    bin_dir: Path, env: dict[str, str], run: Callable[..., Any] = subprocess.run
) -> dict[str, Any]:
    """Version, commit and visible devices from the extracted binary; empty on non-Windows."""
    exe = bin_dir / SERVER_EXE
    out: dict[str, Any] = {"server_version": None, "commit": None, "devices": []}
    for flag, key in (("--version", "version"), ("--list-devices", "devices")):
        try:
            res = run([str(exe), flag], capture_output=True, text=True, env=env, timeout=60)
            text = (res.stdout or "") + (res.stderr or "")
        except (OSError, subprocess.TimeoutExpired):
            continue
        if key == "version":
            out["server_version"], out["commit"] = parse_version_output(text)
        else:
            out["devices"] = parse_devices_output(text)
    return out


def install(settings: Settings, opts: BuildOptions, client: httpx.Client | None = None) -> int:
    backends = [BACKEND_ALIASES.get(b.lower(), b.lower()) for b in (opts.backends or ["both"])]
    if backends == ["both"]:
        backends = list(BACKENDS)
    bad = [b for b in backends if b not in BACKENDS]
    if bad:
        console.print(f"[red]unknown backend[/red] {', '.join(bad)}; use vulkan | rocm | both")
        return 2
    if opts.source not in SOURCES:
        console.print(f"[red]unknown source[/red] {opts.source}; use official | lemonade")
        return 2

    own_client = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))
    platform = HaloPlatform()
    info = load_build_info(settings)
    rc = 0
    try:
        resolved = resolve_tag(client, pinned_commit(), opts.tag)
        if resolved.note:
            console.print(f"[yellow]tag[/yellow] {resolved.tag} ({resolved.note})")
        else:
            console.print(f"[cyan]tag[/cyan] {resolved.tag}")
        for backend in backends:
            source = opts.source if backend == "rocm" else "official"
            existing = info["backends"].get(backend)
            same_request = bool(existing) and (
                existing.get("source") == source
                and (source == "lemonade" or existing.get("tag") == resolved.tag)
            )
            if (
                same_request
                and not opts.force
                and Path(existing.get("bin_dir", "")).joinpath(SERVER_EXE).is_file()
            ):
                console.print(
                    f"[dim]skip[/dim] {backend}: {existing['tag']} installed at "
                    f"{existing['bin_dir']}"
                )
                continue
            repo = LEMONADE_REPO if source == "lemonade" else OFFICIAL_REPO
            if source == "lemonade":
                rel = _get(client, f"{API}/repos/{repo}/releases/latest")
            else:
                rel = _get(client, f"{API}/repos/{repo}/releases/tags/{resolved.tag}")
            tag = rel["tag_name"]
            asset = pick_asset(rel.get("assets", []), ASSET_PATTERNS[(source, backend)], opts.asset)
            console.print(
                f"[cyan]download[/cyan] {asset['name']} ({int(asset.get('size', 0)) >> 20} MB)"
            )
            zip_path = download(
                client,
                asset["browser_download_url"],
                settings.vendor_dir / "downloads" / asset["name"],
            )
            bin_dir = extract(zip_path, settings.vendor_dir / f"{tag}-{backend}")
            entry: dict[str, Any] = {
                "tag": tag,
                "source": source,
                "repo": repo,
                "asset": asset["name"],
                "bin_dir": str(bin_dir),
                "pin_note": resolved.note
                if source == "official"
                else "lemonade tracks its own upstream commit",
            }
            entry.update(probe(bin_dir, platform.runtime_env(settings), run=platform._run))
            pinned = pinned_commit()
            if (
                pinned
                and entry["commit"]
                and not (entry["commit"].startswith(pinned) or pinned.startswith(entry["commit"]))
            ):
                console.print(
                    f"[yellow]note[/yellow] {backend} binary is commit {entry['commit']}, "
                    f"Spark pin is {pinned}; recorded in provenance"
                )
            if backend == "rocm" and entry["devices"] == [] and entry["server_version"]:
                console.print(
                    "[red]rocm build lists no GPU device[/red]: the zip may not include gfx1151 or "
                    "the Adrenalin driver is older than the bundled HIP runtime needs (>= 26.6.4). "
                    "Try: local-llm build --backend rocm --source lemonade --force"
                )
                rc = 1
            info["backends"][backend] = entry
            save_build_info(settings, info)
            console.print(f"[green]installed[/green] {backend} -> {bin_dir}")
    except (httpx.HTTPError, RuntimeError, zipfile.BadZipFile) as exc:
        console.print(f"[red]{exc}[/red]")
        rc = 1
    except KeyError as exc:
        console.print(f"[red]unexpected GitHub API response[/red]: missing {exc}")
        rc = 1
    except OSError as exc:
        console.print(
            f"[red]{exc}[/red]; a running llama-server locks its files on Windows, "
            "run: local-llm stop, then retry"
        )
        rc = 1
    finally:
        if own_client:
            client.close()
    return rc
