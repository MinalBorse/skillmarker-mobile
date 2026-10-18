#!/usr/bin/env python
"""Record a demo of a Streamlit app, driven entirely by demo/scenes.json.

    python scripts/record_demo.py                 # headless
    python scripts/record_demo.py --headed        # watch it happen

Outputs (in demo/output/ by default):
    raw_recording.webm   the screen recording (1920x1080)
    timings.json         when each caption appeared, and which waits to cut
    streamlit.log        the app's console output, for debugging
    FAILED_<scene>.png   only if something went wrong

This file is project-agnostic: only scenes.json changes between projects.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

KNOWN_ACTIONS = ("click", "tab", "select", "type", "scroll_to", "wait_for", "pause")

# ---------------------------------------------------------------------------
# Overlay injected into every page: caption bar + a visible mouse cursor.
#
# The caption text lives in a CSS ::after pseudo-element (via data-text), not in
# the DOM text. That matters: `wait_for` searches the page for result text, and
# if the caption were real text, a caption mentioning "Auto-healed" would satisfy
# `wait_for: "Auto-healed"` instantly, before the app had actually done anything.
#
# Playwright's video doesn't show the real mouse, so we draw our own dot that
# follows mousemove events.
# ---------------------------------------------------------------------------
OVERLAY_JS = r"""
(() => {
  const CSS = `
    #__demo_caption { position: fixed; left: 50%; bottom: 48px; transform: translateX(-50%) translateY(12px);
      max-width: 78vw; padding: 18px 34px; border-radius: 14px; z-index: 2147483647;
      background: rgba(17, 24, 39, 0.88); color: #fff; pointer-events: none;
      font: 600 30px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; text-align: center;
      box-shadow: 0 10px 40px rgba(0,0,0,.35); opacity: 0; transition: opacity .25s ease, transform .25s ease; }
    #__demo_caption.show { opacity: 1; transform: translateX(-50%) translateY(0); }
    #__demo_caption::after { content: attr(data-text); }
    #__demo_cursor { position: fixed; left: 0; top: 0; width: 22px; height: 22px; margin: -11px 0 0 -11px;
      border-radius: 50%; background: rgba(255, 75, 75, .35); border: 2px solid #FF4B4B;
      z-index: 2147483647; pointer-events: none; transition: transform .05s linear, width .1s, height .1s; }
    #__demo_cursor.down { background: rgba(255, 75, 75, .7); }
    .__demo_highlight { outline: 4px solid #FF4B4B !important; outline-offset: 4px !important;
      border-radius: 8px; transition: outline-color .2s; }
  `;
  function ensure() {
    if (!document.body || document.getElementById('__demo_caption')) return;
    const style = document.createElement('style'); style.textContent = CSS; document.head.appendChild(style);
    const cap = document.createElement('div'); cap.id = '__demo_caption'; document.body.appendChild(cap);
    const cur = document.createElement('div'); cur.id = '__demo_cursor'; document.body.appendChild(cur);
    document.addEventListener('mousemove', e => {
      cur.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`; }, true);
    document.addEventListener('mousedown', () => cur.classList.add('down'), true);
    document.addEventListener('mouseup', () => cur.classList.remove('down'), true);
  }
  window.__demoCaption = (text) => {
    ensure();
    const cap = document.getElementById('__demo_caption');
    cap.classList.remove('show');
    void cap.offsetWidth;                 // restart the fade-in animation
    cap.setAttribute('data-text', text);
    if (text) cap.classList.add('show');
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ensure);
  else ensure();
})();
"""


class SceneError(RuntimeError):
    pass


# ---- App process ----------------------------------------------------------------
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(app_cfg: dict, out_dir: Path) -> tuple[subprocess.Popen, str, tempfile.TemporaryDirectory]:
    """Start Streamlit in the background with a fresh temp directory for its data.

    Any env value containing "{tmp}" gets that temp directory, so each recording starts
    from an empty database and is reproducible.
    """
    tmp = tempfile.TemporaryDirectory(prefix="demo_app_", ignore_cleanup_errors=True)  # Windows may still lock the DB file
    env = os.environ.copy()
    for key, value in app_cfg.get("env", {}).items():
        env[key] = str(value).replace("{tmp}", tmp.name)
    env["PYTHONUNBUFFERED"] = "1"

    port = free_port()
    cmd = [sys.executable, "-m", "streamlit", "run", app_cfg.get("entry", "app.py"),
           "--server.port", str(port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none"]
    log = open(out_dir / "streamlit.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"

    deadline = time.monotonic() + app_cfg.get("startup_timeout", 90)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SceneError(f"Streamlit exited early (code {proc.returncode}); see {out_dir / 'streamlit.log'}")
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return proc, url, tmp
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    raise SceneError("Streamlit did not become healthy in time; see streamlit.log")


def stop_app(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---- Recorder -------------------------------------------------------------------
class Recorder:
    def __init__(self, page, settings: dict):
        self.page = page
        self.t0 = time.monotonic()           # the video starts when the page is created
        self.timeout_ms = int(settings.get("action_timeout", 30) * 1000)
        self.cuts: list[dict] = []
        self.scene_id = ""

    def now_ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000)

    # -- syncing with Streamlit ----------------------------------------------------
    def settle(self, timeout_ms: int | None = None):
        """Wait until Streamlit has finished re-running the script.

        After a click, Streamlit keeps showing the *old* page (marked data-stale) while the
        script re-runs. If we searched for result text immediately, we could match stale
        content from the previous run. So: give the rerun a moment to start, then wait for
        the app root to report notRunning and for stale elements to disappear.
        """
        timeout_ms = timeout_ms or self.timeout_ms
        try:
            self.page.wait_for_function(
                "() => document.querySelector('[data-testid=\"stApp\"]')"
                "?.getAttribute('data-test-script-state') === 'running'", timeout=1200)
        except PlaywrightTimeout:
            pass  # the rerun was very fast or no rerun happened
        try:
            self.page.wait_for_function(
                """() => {
                    const app = document.querySelector('[data-testid="stApp"]');
                    const state = app && app.getAttribute('data-test-script-state');
                    const stale = document.querySelectorAll('[data-stale="true"]').length;
                    return state !== 'running' && stale === 0;
                }""", timeout=timeout_ms)
        except PlaywrightTimeout:
            pass  # wait_for still checks the real result text afterwards

    def first_visible(self, candidates, what: str, timeout_ms: int | None = None):
        """Return the first candidate locator that becomes visible (polling all of them)."""
        deadline = time.monotonic() + (timeout_ms or self.timeout_ms) / 1000
        while time.monotonic() < deadline:
            for loc in candidates:
                try:
                    if loc.first.is_visible():
                        return loc.first
                except PlaywrightError:
                    pass
            time.sleep(0.2)
        raise SceneError(f"Could not find {what}")

    # -- visuals -------------------------------------------------------------------
    def caption(self, text: str):
        self.page.evaluate("t => window.__demoCaption && window.__demoCaption(t)", text)

    def point_at(self, loc):
        """Glide the cursor to the element and outline it, so viewers see what happens next."""
        loc.scroll_into_view_if_needed()
        box = loc.bounding_box()
        if box:
            self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=25)
        loc.evaluate("el => el.classList.add('__demo_highlight')")
        self.page.wait_for_timeout(650)

    def unhighlight(self, loc):
        try:
            loc.evaluate("el => el.classList.remove('__demo_highlight')")
        except PlaywrightError:
            pass  # the element was replaced by a rerun, so the outline is gone anyway

    # -- actions ---------------------------------------------------------------------
    def do_click(self, label: str):
        p = self.page
        exact = re.compile(rf"^\s*{re.escape(label)}\s*$")
        loc = self.first_visible([
            p.get_by_role("button", name=label, exact=True),
            p.get_by_role("button", name=label),
            p.locator("summary").filter(has_text=label),  # st.expander headers
            p.get_by_role("tab", name=label),
            p.get_by_role("link", name=label),
            p.get_by_text(exact).filter(visible=True),
        ], f"button '{label}'")
        self.point_at(loc)
        loc.click()
        self.unhighlight(loc)
        self.settle()

    def do_tab(self, label: str):
        """Tabs: st.tabs, or a sidebar st.radio used as page navigation."""
        p = self.page
        exact = re.compile(rf"^\s*{re.escape(label)}\s*$")
        loc = self.first_visible([
            p.get_by_role("tab", name=label),
            p.locator('[data-testid="stRadio"] label').filter(has_text=exact),
            p.locator('[data-testid="stPageLink-NavLink"]').filter(has_text=exact),
            p.locator('[data-testid="stSidebarNav"] a').filter(has_text=exact),
        ], f"tab '{label}'")
        self.point_at(loc)
        loc.click()
        self.unhighlight(loc)
        self.settle()

    def do_select(self, label: str, option: str):
        """Open a dropdown, filter by typing part of the option, click the first match.

        Option labels are often built from data ("#12: Schema change", emojis, arrows),
        so exact text is brittle. Typing a fragment uses the dropdown's own search.
        """
        p = self.page
        box = self.first_visible([
            p.locator('[data-testid="stSelectbox"]').filter(has=p.get_by_text(label, exact=True)),
            p.locator('[data-testid="stMultiSelect"]').filter(has=p.get_by_text(label, exact=True)),
            p.locator('[data-testid="stSelectbox"]').filter(has_text=label),
        ], f"dropdown '{label}'")
        self.point_at(box)
        # Newer Streamlit renders a react-aria combobox; older versions a baseweb select.
        field = self.first_visible([
            box.get_by_role("combobox"),
            box.locator('[data-baseweb="select"] input'),
            box.locator("input"),
        ], f"input of dropdown '{label}'", timeout_ms=5000)
        field.click()
        field.fill("")                 # clear the current value so typing filters from scratch
        p.keyboard.type(option, delay=70)
        choice = self.first_visible([
            p.get_by_role("option").filter(has_text=option),
            p.get_by_role("option"),
        ], f"option matching '{option}' in '{label}'", timeout_ms=8000)
        p.wait_for_timeout(400)
        choice.click()
        self.unhighlight(box)
        self.settle()

    def do_type(self, text: str, into: str | None, submit: bool):
        """Type into a labelled text input, or into the chat input if no label is given."""
        p = self.page
        if into:
            candidates = [p.get_by_label(into, exact=True), p.get_by_placeholder(into), p.get_by_label(into)]
        else:
            candidates = [p.locator('[data-testid="stChatInputTextArea"]'),
                          p.locator('[data-testid="stChatInput"] textarea')]
        field = self.first_visible(candidates, f"text input '{into or 'chat input'}'")
        self.point_at(field)
        field.click()
        field.fill("")
        field.press_sequentially(text, delay=45)   # visible typing looks natural on video
        if submit:
            field.press("Enter")
        self.unhighlight(field)
        self.settle()

    def do_scroll_to(self, text: str):
        # Only visible matches: Streamlit can keep hidden copies of text in the page.
        loc = self.first_visible([self.page.get_by_text(text).filter(visible=True)], f"text '{text}' to scroll to")
        loc.evaluate("el => el.scrollIntoView({behavior: 'smooth', block: 'center'})")
        self.page.wait_for_timeout(900)  # let the smooth scroll finish on camera

    def do_wait_for(self, text: str, timeout_s: float | None, cut: bool):
        """Wait for result text to actually appear, never for a fixed time.

        AI calls take 2 seconds on a good day and 40 on a bad one. Waiting for the
        result keeps every scene in sync with what is on screen. Long waits can be
        marked `cut: true`; we record their start and end so add_voiceover.py can
        remove the dead time from the video.
        """
        timeout_ms = int((timeout_s or self.timeout_ms / 1000) * 1000)
        start = self.now_ms()
        self.settle(timeout_ms)
        remaining = max(1000, timeout_ms - (self.now_ms() - start))
        try:
            self.page.get_by_text(text).filter(visible=True).first.wait_for(state="visible", timeout=remaining)
        except PlaywrightTimeout as e:
            raise SceneError(f"Text '{text}' did not appear within {timeout_ms / 1000:.0f}s") from e
        end = self.now_ms()
        if cut:
            self.cuts.append({"scene": self.scene_id, "start_ms": start, "end_ms": end})
        return (end - start) if cut else 0

    def run_action(self, action: dict) -> int:
        """Perform one action. Returns how many ms of it will be cut from the video."""
        if "click" in action:
            self.do_click(action["click"])
        elif "tab" in action:
            self.do_tab(action["tab"])
        elif "select" in action:
            self.do_select(action["select"], action["option"])
        elif "type" in action:
            self.do_type(action["type"], action.get("into"), action.get("submit", True))
        elif "scroll_to" in action:
            self.do_scroll_to(action["scroll_to"])
        elif "wait_for" in action:
            return self.do_wait_for(action["wait_for"], action.get("timeout"), action.get("cut", False))
        elif "pause" in action:
            self.page.wait_for_timeout(int(action["pause"]))   # pacing only, never for syncing
        else:
            raise SceneError(f"Unknown action {action}; expected one of {KNOWN_ACTIONS}")
        return 0


def narration_seconds(text: str, words_per_second: float) -> float:
    return len(text.split()) / words_per_second + 0.6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="demo/scenes.json")
    ap.add_argument("--out", default="demo/output")
    ap.add_argument("--headed", action="store_true", help="show the browser while recording")
    args = ap.parse_args()

    cfg = json.loads(Path(args.scenes).read_text(encoding="utf-8"))
    app_cfg, settings, scenes = cfg.get("app", {}), cfg.get("settings", {}), cfg["scenes"]
    vp = app_cfg.get("viewport", {"width": 1920, "height": 1080})
    wps = float(settings.get("words_per_second", 2.6))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("FAILED_*.png"):
        old.unlink()

    print(f"Starting {app_cfg.get('entry', 'app.py')} ...")
    proc, url, tmp = start_app(app_cfg, out)
    timings = {"video": "raw_recording.webm", "viewport": vp, "scenes": [], "cuts": []}
    failed = None
    try:
        with sync_playwright() as pw, tempfile.TemporaryDirectory() as video_dir:
            browser = pw.chromium.launch(headless=not args.headed)
            context = browser.new_context(viewport=vp, record_video_dir=video_dir,
                                          record_video_size=vp, device_scale_factor=1)
            context.add_init_script(OVERLAY_JS)
            page = context.new_page()
            rec = Recorder(page, settings)
            try:
                page.goto(url)
                ready = app_cfg.get("ready_text")
                if ready:
                    page.get_by_text(ready).first.wait_for(timeout=app_cfg.get("startup_timeout", 90) * 1000)
                rec.settle()
                timings["ready_ms"] = rec.now_ms()
                page.wait_for_timeout(600)

                for i, scene in enumerate(scenes):
                    rec.scene_id = scene.get("id", f"scene{i + 1}")
                    print(f"[{i + 1}/{len(scenes)}] {rec.scene_id}: {scene['caption']}")
                    rec.caption(scene["caption"])
                    entry = {"index": i, "id": rec.scene_id, "caption": scene["caption"],
                             "caption_ms": rec.now_ms()}
                    timings["scenes"].append(entry)
                    scene_start, cut_ms = time.monotonic(), 0
                    for action in scene.get("actions", []):
                        cut_ms += rec.run_action(action)

                    # Keep the scene on screen long enough for its narration. Time that will be
                    # cut doesn't count, because viewers won't see it.
                    needed = max(float(scene.get("hold", 0)), narration_seconds(scene.get("voice", ""), wps))
                    shown = time.monotonic() - scene_start - cut_ms / 1000
                    if shown < needed:
                        page.wait_for_timeout(int((needed - shown) * 1000))
                    entry["end_ms"] = rec.now_ms()

                rec.caption("")
                page.wait_for_timeout(int(settings.get("end_hold_ms", 2500)))
            except (SceneError, PlaywrightError, KeyError) as e:
                failed = rec.scene_id or "startup"
                shot = out / f"FAILED_{failed}.png"
                page.screenshot(path=str(shot), full_page=False)
                print(f"\nFAILED in scene '{failed}': {e}\nScreenshot: {shot}", file=sys.stderr)
            finally:
                timings["end_ms"] = rec.now_ms()
                timings["cuts"] = rec.cuts
                video = page.video
                context.close()            # the video file is finalized when the context closes
                video.save_as(out / "raw_recording.webm")
                browser.close()
    finally:
        stop_app(proc)
        tmp.cleanup()

    timings["failed_scene"] = failed
    (out / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    if failed:
        return 1
    cut_s = sum(c["end_ms"] - c["start_ms"] for c in timings["cuts"]) / 1000
    print(f"\nSaved {out / 'raw_recording.webm'} ({timings['end_ms'] / 1000:.0f}s, "
          f"{cut_s:.0f}s of AI waits marked for cutting) and timings.json")
    print("Next: python scripts/add_voiceover.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
