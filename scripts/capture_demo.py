"""Record the README hero GIF and screenshots from the running app with headless Chromium.

Selection rule (fixed in advance, recorded in ``demo.json``): walk the "lie" presets in order and
the truth-serum strengths from ``steering.json`` in increasing order (keeping only strengths whose
QA accuracy stays within 10 points of the unsteered model), and use the first preset whose
unsteered reply the probe flags and whose steered reply it does not. If no pair qualifies, the
steered reply with the lowest score is recorded instead and ``demo.json`` says ``rule_met:
false``. All generation is greedy, so the recorded run reproduces exactly what the API returned.

    uv run python scripts/capture_demo.py --url http://127.0.0.1:8096 --results results/qwen3-4b
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image
from playwright.sync_api import Page, sync_playwright

LIE_PRESETS = ("car", "password", "work", "insider", "allergy")


def chat(url: str, messages: list[dict[str, str]], alpha: float) -> dict[str, Any]:
    body = json.dumps({"messages": messages, "truth_serum": alpha, "temperature": 0}).encode()
    req = urllib.request.Request(
        f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as res:
        events = [json.loads(line) for line in res.read().decode().splitlines() if line.strip()]
    text = "".join(e["text"] for e in events if e["type"] == "token")
    done = events[-1]
    if done["type"] != "done":
        raise RuntimeError(f"generation failed: {done}")
    return {"text": text, "p": done["p_response"], "flagged": done["flagged"]}


def candidate_alphas(results: Path) -> list[float]:
    rows = json.loads((results / "steering.json").read_text())["rows"]
    base = next(r["qa_accuracy"] for r in rows if r["alpha"] == 0)
    return [r["alpha"] for r in rows if r["alpha"] > 0 and r["qa_accuracy"] >= base - 0.10]


def choose(url: str, results: Path) -> dict[str, Any]:
    """Apply the selection rule. If no pair meets it, fall back to the steered reply with the
    lowest probe score (``rule_met: false``), so the recording never overstates the serum."""
    presets = {p["id"]: p for p in json.loads(urllib.request.urlopen(f"{url}/api/presets").read())}
    tried = []
    for pid in LIE_PRESETS:
        p = presets[pid]
        msgs = [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]
        base = chat(url, msgs, 0.0)
        tried.append({"preset": pid, "alpha": 0.0, **base})
        if not base["flagged"]:
            continue
        for alpha in candidate_alphas(results):
            steered = chat(url, msgs, alpha)
            tried.append({"preset": pid, "alpha": alpha, **steered})
            if not steered["flagged"]:
                return {"preset": pid, "alpha": alpha, "rule_met": True, "tried": tried}
    steered_runs = [t for t in tried if t["alpha"] > 0]
    if not steered_runs:
        return {"preset": None, "alpha": None, "rule_met": False, "tried": tried}
    best = min(steered_runs, key=lambda t: t["p"])
    return {"preset": best["preset"], "alpha": best["alpha"], "rule_met": False, "tried": tried}


class Recorder:
    """Collects downscaled screenshots; consecutive duplicates are merged into one longer frame."""

    def __init__(self, page: Page, width: int) -> None:
        self.page, self.width = page, width
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def snap(self, ms: int) -> None:
        img = Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        img = img.resize((self.width, round(img.height * self.width / img.width)), Image.LANCZOS)
        if self.frames and img.tobytes() == self.frames[-1].tobytes():
            self.durations[-1] += ms
            return
        self.frames.append(img)
        self.durations.append(ms)

    def until_idle(self, every_ms: int = 160, timeout_s: float = 240) -> None:
        """Snap while the model streams, until the Send button is enabled again."""
        start = time.time()
        time.sleep(0.3)
        while time.time() - start < timeout_s:
            self.snap(every_ms)
            if self.page.locator("#send-btn").is_enabled():
                return
            time.sleep(every_ms / 1000)
        raise TimeoutError("generation did not finish")

    def save(self, path: Path, max_bytes: int = 7_500_000) -> None:
        """Write the GIF, shrinking the frame width until it fits under ``max_bytes``."""
        for width in (1080, 960, 840, 720):
            frames = [
                f.resize((width, round(f.height * width / f.width)), Image.LANCZOS).quantize(
                    colors=128, method=Image.Quantize.MEDIANCUT
                )
                for f in self.frames
            ]
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=self.durations,
                loop=0,
                optimize=True,
                disposal=1,
            )
            if path.stat().st_size <= max_bytes:
                return
        raise RuntimeError(f"{path} is still over {max_bytes} bytes at 720 px wide")


def record(url: str, out: Path, preset: str, alpha: float) -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 860})
        page.goto(url)
        page.wait_for_selector(".preset")
        rec = Recorder(page, 1080)
        rec.snap(900)
        page.click(f'.preset[data-id="{preset}"]')
        rec.until_idle()
        rec.snap(2200)
        page.screenshot(path=str(out / "polygraph.png"))
        # Turn the truth serum up visibly, then regenerate the same turn.
        slider = page.locator("#serum")
        for step in range(1, 7):
            value = alpha * step / 6
            slider.evaluate(
                "(el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }", value
            )
            rec.snap(140)
        rec.snap(700)
        page.click("#regen-btn")
        rec.until_idle()
        rec.snap(3000)
        page.screenshot(path=str(out / "truth_serum.png"))
        rec.save(out / "demo.gif")
        capture_research(page, out / "research.png")
        browser.close()


def capture_research(page: Page, path: Path) -> None:
    """Screenshot the whole research tab.

    The tab scrolls inside its own viewport-high container, which a full-page screenshot cannot
    see past, so the container's height is unpinned first and the page grows to fit it.
    """
    page.click('.tab[data-tab="research"]')
    page.wait_for_timeout(1200)
    page.evaluate(
        "() => { const r = document.querySelector('#research');"
        " r.style.height = 'auto'; r.style.overflow = 'visible'; }"
    )
    page.wait_for_timeout(300)
    page.screenshot(path=str(path), full_page=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8096")
    ap.add_argument("--results", type=Path, default=Path("results/qwen3-4b"))
    ap.add_argument("--out", type=Path, default=Path("docs/assets"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    choice = choose(args.url, args.results)
    (args.out / "demo.json").write_text(json.dumps(choice, indent=2) + "\n")
    if choice["preset"] is None:
        raise SystemExit("no lie preset was flagged, so there is nothing to steer; see demo.json")
    record(args.url, args.out, choice["preset"], choice["alpha"])
    print(json.dumps({k: v for k, v in choice.items() if k != "tried"}))


if __name__ == "__main__":
    main()
