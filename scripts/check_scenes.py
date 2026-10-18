#!/usr/bin/env python
"""Check demo/scenes.json against the app's source code before recording.

    python scripts/check_scenes.py

A recording takes minutes; a typo in a button label wastes all of them. This script
reads every string literal in the project's Python files and verifies that each
button, tab, dropdown label, dropdown option and input label used in scenes.json
appears in the code. It also validates the file's structure and estimates the
narration length.

Exit code 1 if anything is wrong. This file is project-agnostic.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

KNOWN_ACTIONS = {"click", "tab", "select", "type", "scroll_to", "wait_for", "pause"}
ACTION_OPTIONS = {"option", "into", "submit", "timeout", "cut"}
SKIP_DIRS = {".venv", "venv", "env", ".git", "node_modules", "__pycache__", "site-packages",
             "demo", "scripts", "tests", "build", "dist", ".pytest_cache"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def collect_strings(root: Path) -> set[str]:
    """All string literals in the project's Python files, including the fixed parts of f-strings.

    UI labels are almost always literals (st.button("Approve fix"), dict keys used to build
    buttons, CHECK_LABELS used to build dropdown options), so a substring match against
    them catches typos without having to run the app.
    """
    found = set()
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  (skipped {path}: {e})")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(norm(node.value))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="demo/scenes.json")
    ap.add_argument("--root", default=".", help="project root containing the app code")
    args = ap.parse_args()

    errors, warnings = [], []
    try:
        cfg = json.loads(Path(args.scenes).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Cannot read {args.scenes}: {e}")
        return 1

    root = Path(args.root)
    app = cfg.get("app", {})
    entry = root / app.get("entry", "app.py")
    if not entry.exists():
        errors.append(f"app.entry '{entry}' does not exist")
    if not app.get("ready_text"):
        warnings.append("app.ready_text is not set; recording starts without waiting for the app to load")
    if not any("{tmp}" in str(v) for v in app.get("env", {}).values()):
        warnings.append("no app.env value uses {tmp}; the recording may reuse your real database")

    strings = collect_strings(root)
    print(f"Scanned {len(strings)} string literals in {root.resolve()}\n")

    # (kind, value, scene, must_exist)
    targets = []
    scenes = cfg.get("scenes", [])
    if not scenes:
        errors.append("no scenes defined")
    seen_ids, words_per_second = set(), float(cfg.get("settings", {}).get("words_per_second", 2.6))
    total_words = 0

    for i, scene in enumerate(scenes, 1):
        sid = scene.get("id", f"scene{i}")
        where = f"scene {i} ({sid})"
        if sid in seen_ids:
            errors.append(f"{where}: duplicate id")
        seen_ids.add(sid)
        for field in ("caption", "voice"):
            if not str(scene.get(field, "")).strip():
                errors.append(f"{where}: missing '{field}'")
        words = len(str(scene.get("voice", "")).split())
        total_words += words
        if len(scene.get("caption", "")) > 90:
            warnings.append(f"{where}: caption is {len(scene['caption'])} chars; it may wrap to 3 lines")

        for j, action in enumerate(scene.get("actions", []), 1):
            at = f"{where}, action {j}"
            kinds = [k for k in action if k in KNOWN_ACTIONS]
            unknown = [k for k in action if k not in KNOWN_ACTIONS | ACTION_OPTIONS]
            if len(kinds) != 1:
                errors.append(f"{at}: needs exactly one of {sorted(KNOWN_ACTIONS)}, got {list(action)}")
                continue
            if unknown:
                errors.append(f"{at}: unknown key(s) {unknown}")
            kind = kinds[0]
            value = action[kind]
            if action.get("cut") and kind != "wait_for":
                errors.append(f"{at}: 'cut' only works on wait_for")
            if "timeout" in action and not isinstance(action["timeout"], (int, float)):
                errors.append(f"{at}: timeout must be a number of seconds")
            if kind == "click":
                targets.append(("button", value, at, True))
            elif kind == "tab":
                targets.append(("tab", value, at, True))
            elif kind == "select":
                targets.append(("dropdown label", value, at, True))
                if "option" not in action:
                    errors.append(f"{at}: select needs an 'option' (part of the option text)")
                else:
                    targets.append(("dropdown option", action["option"], at, True))
            elif kind == "type":
                if action.get("into"):
                    targets.append(("input label", action["into"], at, True))
            elif kind in ("wait_for", "scroll_to"):
                # Result text can come from data or the AI at runtime, so only warn.
                targets.append((kind, value, at, False))
            elif kind == "pause" and not isinstance(value, (int, float)):
                errors.append(f"{at}: pause must be milliseconds")

    width = max((len(str(t[1])) for t in targets), default=10)
    for kind, value, at, must in targets:
        ok = any(norm(value) in s for s in strings)
        mark = "ok " if ok else ("ERR" if must else "?  ")
        print(f"  {mark} {kind:16s} {str(value):{width}s}  {at}")
        if not ok:
            msg = f"{at}: {kind} '{value}' not found in the app code"
            (errors if must else warnings).append(msg + ("" if must else " (fine if it comes from data/AI)"))

    est = total_words / words_per_second + 0.6 * len(scenes)
    print(f"\nNarration: {total_words} words, about {est / 60:.1f} min of speech across {len(scenes)} scenes")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("\nErrors (fix these before recording):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll good. Next: python scripts/record_demo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
