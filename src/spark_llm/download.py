"""Download GGUF artifacts from Hugging Face into models_dir."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from spark_llm.config import Settings, get_settings
from spark_llm.console import err as console
from spark_llm.registry import ModelKind, ModelSpec


def shard_base(name: str) -> str:
    """Strip -NNNNN-of-MMMMM from a GGUF filename stem for sibling matching."""
    stem = name.removesuffix(".gguf")
    parts = stem.rsplit("-", 3)
    if len(parts) >= 4 and parts[-2] == "of" and parts[-1].isdigit() and parts[-3].isdigit():
        return "-".join(parts[:-3])
    return stem


def is_shard_sibling(filename: str, primary: str) -> bool:
    """True if filename is the primary GGUF or a matching multi-shard sibling."""
    if Path(filename).name == Path(primary).name:
        return True
    if not filename.endswith(".gguf"):
        return False
    primary_name = Path(primary).name
    if "-of-" in primary_name and "-of-" in filename:
        return shard_base(Path(filename).name) == shard_base(primary_name)
    return False


def resolve_local(spec: ModelSpec, models_dir: Path) -> Path | None:
    """Return existing local path for the primary GGUF if present."""
    if not spec.file:
        return None
    direct = models_dir / spec.file
    if direct.is_file():
        return direct
    matches = list(models_dir.rglob(spec.file))
    return matches[0] if matches else None


def resolve_weights(spec: ModelSpec, models_dir: Path) -> Path | None:
    """Return the local primary GGUF for a spec, whether declared by ``file`` or ``repo``/``quant``.

    Mirrors the two storage layouts ``download_model`` produces: an explicit ``file`` under
    ``models_dir`` (possibly nested), or a ``<org>__<repo>`` snapshot directory filtered by
    quant. Returns None when nothing is on disk.
    """
    if spec.file:
        return resolve_local(spec, models_dir)
    snap = spec.snapshot_dir(models_dir)
    if snap is None or not snap.is_dir():
        return None
    ggufs = sorted(snap.rglob("*.gguf"))
    if spec.quant:
        filtered = [g for g in ggufs if spec.quant.lower() in g.name.lower()]
        ggufs = filtered or ggufs
    if not ggufs:
        return None
    # Multi-shard repos: point at the first shard; llama.cpp loads siblings.
    first = [g for g in ggufs if "-00001-of-" in g.name]
    return first[0] if first else ggufs[0]


def download_model(spec: ModelSpec, settings: Settings | None = None) -> Path:
    """Download model weights into settings.models_dir; return primary GGUF path."""
    settings = settings or get_settings()
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    existing = resolve_weights(spec, models_dir)
    if existing is not None:
        console.print(f"[green]already present[/green] {existing}")
        return existing

    if spec.file and spec.repo:
        console.print(f"[cyan]downloading[/cyan] {spec.repo} → {models_dir}")
        files = list_repo_files(spec.repo)
        wanted = [f for f in files if is_shard_sibling(f, spec.file)]
        if not wanted:
            wanted = [f for f in files if Path(f).name == spec.file]
        if not wanted:
            raise FileNotFoundError(f"could not find {spec.file!r} (or shards) in {spec.repo}")
        for remote in wanted:
            local = hf_hub_download(
                repo_id=spec.repo,
                filename=remote,
                local_dir=str(models_dir),
            )
            console.print(f"  got {local}")
        resolved = resolve_local(spec, models_dir)
        if resolved is None:
            raise FileNotFoundError(
                f"download finished but {spec.file} not found under {models_dir}"
            )
        return resolved

    if spec.repo:
        console.print(
            f"[cyan]snapshot[/cyan] {spec.repo}"
            + (f":{spec.quant}" if spec.quant else "")
            + f" → {models_dir}"
        )
        local_dir = models_dir / spec.repo.replace("/", "__")
        allow: list[str] | None = ["*.gguf"]
        if spec.quant:
            allow = [f"*{spec.quant}*.gguf", f"*{spec.quant.lower()}*.gguf"]
        path = snapshot_download(
            repo_id=spec.repo,
            local_dir=str(local_dir),
            allow_patterns=allow,
        )
        root = Path(path)
        ggufs = sorted(root.rglob("*.gguf"))
        if spec.quant:
            filtered = [g for g in ggufs if spec.quant.lower() in g.name.lower()]
            ggufs = filtered or ggufs
        if not ggufs:
            raise FileNotFoundError(f"no GGUF files found under {root}")
        console.print(f"[green]downloaded[/green] {ggufs[0]}")
        return ggufs[0]

    raise ValueError(
        f"model {spec.name!r} has nothing to download "
        f"(need file + repo, or repo, or a local file already present)"
    )


def download_hf_ref(hf: str, settings: Settings | None = None) -> Path:
    """Resolve an org/repo:quant (or org/repo) ref via huggingface_hub.

    Used for the CLI ``--hf`` escape hatch so we do not depend on llama-server
    being linked with OpenSSL/HTTPS (common gap on minimal Spark images).
    """
    settings = settings or get_settings()
    repo, _, quant = hf.partition(":")
    if not repo:
        raise ValueError(f"invalid --hf value: {hf!r}")
    anon = ModelSpec(
        name=hf.replace("/", "_").replace(":", "_"),
        repo=repo,
        quant=quant or None,
        kind=ModelKind.chat,
    )
    return download_model(anon, settings)
