# Deck: local inference results (September 2026)

Talk-through slides for the team. Words on the slides are minimal; speaker cues are HTML
comments in the Marp source. Every number traces to
[`../eval-report-2026-09-spark-halo-terra.md`](../eval-report-2026-09-spark-halo-terra.md)
and the CI column of [`../../state/evals/report-sec.md`](../../state/evals/report-sec.md).

| File | What |
| --- | --- |
| `local-inference-2026-09.md` | Marp source (edit this) |
| `make_charts.py` | regenerates `charts/*.png` from the numbers at the top of the script |
| `charts/` | `accuracy.png`, `prefill-decode.png` |
| `local-inference-2026-09.html` | rendered deck; open in any browser, press F for fullscreen |
| `local-inference-2026-09.pdf` | rendered deck, when a browser was available at render time |

## Regenerate

```bash
uv run --with matplotlib python docs/deck/make_charts.py
npx @marp-team/marp-cli docs/deck/local-inference-2026-09.md -o docs/deck/local-inference-2026-09.html   # add --offline if npx stalls resolving @latest
# PDF: Marp's own --pdf cannot drive a Windows browser from WSL, but Chrome headless can
# print the rendered HTML directly (one slide per page):
WIN=$(wslpath -w "$PWD/docs/deck/local-inference-2026-09.html")
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu \
  --no-pdf-header-footer --print-to-pdf="$(cmd.exe /c 'echo %TEMP%' | tr -d '\r')\\deck.pdf" "file:///$WIN"
cp "$(wslpath -u "$(cmd.exe /c 'echo %TEMP%' | tr -d '\r')")/deck.pdf" docs/deck/local-inference-2026-09.pdf
```

On a machine with a local Chromium, `npx @marp-team/marp-cli … --pdf --allow-local-files` works
directly. If neither is available, open the HTML in a browser and print to PDF.

Charts follow the repo's data-viz rules: one series, one hue, direct value labels in text ink,
two panels rather than a dual axis, light surface only (the deck is projected; no dark variant).
