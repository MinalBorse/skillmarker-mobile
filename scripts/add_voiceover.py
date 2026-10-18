#!/usr/bin/env python
"""Turn raw_recording.webm + timings.json into narrated, ready-to-publish videos.

    python scripts/add_voiceover.py
    python scripts/add_voiceover.py --voice en-GB-SoniaNeural --rate +5%
    edge-tts --list-voices            # see all available voices

Outputs (in demo/output/):
    demo_youtube.mp4    1080p, high quality
    demo_github.mp4     re-encoded to stay under the GitHub README limit (10 MB)
    subtitles.srt       narration subtitles matching demo_youtube.mp4

Steps: text-to-speech per scene -> cut long AI waits (one streaming ffmpeg pass)
-> place each clip at its caption's timestamp -> mix over a silent base track -> export.

This file is project-agnostic: only scenes.json changes between projects.
"""
import argparse
import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

FPS = 25                    # Playwright records at 25 fps; we normalize to constant 25 fps
VOICE_LEAD_S = 0.25         # start talking just after the caption fades in
GAP_S = 0.20                # minimum silence between two clips
FADE_S = 0.15


# ---- ffmpeg discovery ---------------------------------------------------------------
def find_ffmpeg() -> tuple[str, str]:
    """Use the system ffmpeg if there is one, otherwise download a static build.

    static-ffmpeg fetches a pinned ffmpeg binary on first use and adds it to PATH,
    so nobody has to install ffmpeg by hand (especially convenient on Windows).
    """
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    try:
        import static_ffmpeg
    except ImportError:
        sys.exit("ffmpeg not found. Install it, or run: pip install static-ffmpeg")
    print("No system ffmpeg found; using static-ffmpeg (downloads once on first use) ...")
    try:
        static_ffmpeg.add_paths()
    except Exception as e:  # usually no internet on the first run
        sys.exit(f"static-ffmpeg could not download ffmpeg ({type(e).__name__}). "
                 "Check your internet connection, or install ffmpeg yourself.")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not (ffmpeg and ffprobe):
        sys.exit("static-ffmpeg could not provide ffmpeg/ffprobe")
    return ffmpeg, ffprobe


def run(cmd: list[str], what: str):
    print(f"  ffmpeg: {what} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        sys.exit(f"ffmpeg failed while trying to {what}")


def duration(ffprobe: str, path: Path) -> float:
    out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip())


# ---- 1. Text to speech --------------------------------------------------------------
async def _speak(text: str, voice: str, rate: str, path: Path):
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate).save(str(path))


def synthesize(scenes: list[dict], voice: str, rate: str, work: Path) -> list[Path | None]:
    """One mp3 per scene. Clips are cached by (text, voice, rate), so re-running after
    changing only timings or export settings doesn't call the TTS service again."""
    paths, jobs = [], []
    for i, scene in enumerate(scenes):
        text = scene.get("voice", "").strip()
        if not text:
            paths.append(None)
            continue
        key = hashlib.sha1(f"{text}|{voice}|{rate}".encode()).hexdigest()[:10]
        path = work / f"voice_{i:02d}_{key}.mp3"
        paths.append(path)
        if not path.exists() or path.stat().st_size == 0:
            jobs.append(_speak(text, voice, rate, path))

    async def go():
        await asyncio.gather(*jobs)

    if jobs:
        print(f"Generating {len(jobs)} voice clip(s) with {voice} ({rate}) ...")
        try:
            asyncio.run(go())
        except Exception as e:  # network down, unknown voice name, service hiccup
            for p in paths:
                if p and p.exists() and p.stat().st_size == 0:
                    p.unlink()  # don't cache a failed (empty) clip
            sys.exit(f"Text-to-speech failed ({type(e).__name__}: {e}).\n"
                     "edge-tts needs an internet connection. Check the voice name with: edge-tts --list-voices")
    return paths


# ---- 2. Cutting the long waits ---------------------------------------------------------
def cut_ranges(timings: dict, keep_s: float, min_cut_s: float) -> list[tuple[float, float]]:
    """Time ranges (seconds, original recording) to remove.

    - The start-up before the first caption (blank page, app loading).
    - Each `cut: true` wait, minus `keep_s` at both ends, so viewers still see the click
      and a moment of the spinner, then the result. Short waits aren't worth a jump cut.
    """
    ranges = []
    if timings["scenes"]:
        head = timings["scenes"][0]["caption_ms"] / 1000 - 0.4
        if head > 0.5:
            ranges.append((0.0, head))
    for c in timings.get("cuts", []):
        a, b = c["start_ms"] / 1000 + keep_s, c["end_ms"] / 1000 - keep_s
        if b - a >= min_cut_s:
            ranges.append((a, b))
    ranges.sort()
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def to_output_time(t: float, ranges: list[tuple[float, float]]) -> float:
    """Map a time in the raw recording to the same moment in the cut video."""
    removed = sum(max(0.0, min(t, b) - a) for a, b in ranges if a < t)
    return t - removed


def cut_video(ffmpeg: str, raw: Path, ranges, size: tuple[int, int], out: Path):
    """Remove all cut ranges in ONE streaming ffmpeg pass.

    The usual approach (trim each kept segment, then concat) makes ffmpeg buffer the
    segments' frames in memory; at 1080p that can exhaust 8 GB of RAM. Here every frame
    flows through once: `fps` makes the variable-rate webm a constant 25 fps, `select`
    drops frames inside the cut ranges, and `setpts` renumbers the survivors so the video
    plays continuously. Memory use stays flat no matter how long the recording is.
    """
    w, h = size
    filters = [f"fps={FPS}"]
    if ranges:
        expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in ranges)
        filters.append(f"select='not({expr})'")
    filters += [f"setpts=N/{FPS}/TB",
                f"scale={w}:{h}:force_original_aspect_ratio=decrease",
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=white",
                "format=yuv420p"]
    run([ffmpeg, "-y", "-hide_banner", "-i", str(raw), "-vf", ",".join(filters), "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(out)],
        f"cut {len(ranges)} range(s) from the recording")


# ---- 3. Placing the voice clips --------------------------------------------------------
def plan_clips(timings: dict, ranges, clip_paths, clip_durs, video_s: float, max_speed: float) -> list[dict]:
    """Decide when each clip starts and how fast it plays.

    Each clip starts when its caption appeared (mapped into the cut video). A clip must
    end before the next caption, or two voices would overlap. If it's too long, we speed
    up only that clip, but never beyond `max_speed` (1.35x by default): faster than that,
    neural voices start to sound rushed and unnatural. If even that isn't enough, the clip
    is faded out at the boundary and a warning tells you which scene to shorten.
    """
    starts = [to_output_time(s["caption_ms"] / 1000, ranges) for s in timings["scenes"]]
    plan = []
    for i, (path, dur) in enumerate(zip(clip_paths, clip_durs)):
        if path is None or i >= len(starts):
            continue
        start = starts[i] + VOICE_LEAD_S
        end_limit = (starts[i + 1] - GAP_S) if i + 1 < len(starts) else (video_s - 0.3)
        slot = max(0.5, end_limit - start)
        speed, play = 1.0, dur
        if dur > slot:
            speed = min(max_speed, dur / slot)
            play = dur / speed
        truncated = play > slot + 0.01
        if speed > 1.0:
            note = f"sped up {speed:.2f}x"
            if truncated:
                note += f", still {play - slot:.1f}s too long: faded out (shorten this scene's voice text)"
            print(f"  scene {i + 1} ({timings['scenes'][i]['id']}): {note}")
        plan.append({"index": i, "path": path, "start": start, "speed": speed,
                     "play": min(play, slot), "truncated": truncated})
    return plan


def mix_and_export(ffmpeg: str, video: Path, plan: list[dict], video_s: float, out: Path):
    """Mix all clips over a silent base track and mux with the cut video (video is copied).

    Why the silent base track: if the audio stream only began at the first clip's
    delay, the audio would start at e.g. 0.6s while video starts at 0. Many players
    (and MP4 edit lists) then align the first audio sample with t=0, so the narration
    plays early and drifts from the captions. A silent track from 0 to the end makes the
    audio stream start at exactly 0 and last exactly as long as the video.
    """
    cmd = [ffmpeg, "-y", "-hide_banner", "-i", str(video),
           "-f", "lavfi", "-t", f"{video_s:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    parts, labels = [], []
    for k, clip in enumerate(plan):
        cmd += ["-i", str(clip["path"])]
        chain = [f"[{k + 2}:a]aresample=48000", "aformat=channel_layouts=stereo"]
        if clip["speed"] > 1.0:
            chain.append(f"atempo={clip['speed']:.4f}")
        if clip["truncated"]:
            chain += [f"atrim=0:{clip['play']:.3f}",
                      f"afade=t=out:st={max(0, clip['play'] - FADE_S):.3f}:d={FADE_S}"]
        delay_ms = int(round(clip["start"] * 1000))
        chain.append(f"adelay={delay_ms}:all=1")
        parts.append(",".join(chain) + f"[v{k}]")
        labels.append(f"[v{k}]")
    n = len(plan) + 1
    parts.append(f"[1:a]{''.join(labels)}amix=inputs={n}:duration=first:dropout_transition=0:normalize=0,"
                 f"alimiter=limit=0.95[aout]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{video_s:.3f}",
            "-movflags", "+faststart", str(out)]
    run(cmd, f"mix {len(plan)} voice clip(s) and export {out.name}")


def export_github(ffmpeg: str, src: Path, video_s: float, out: Path, limit_mb: float, work: Path):
    """Re-encode to fit under the README upload limit, using two-pass x264 for accurate size."""
    target_bits = limit_mb * 1_000_000 * 8 * 0.96          # 4% safety margin for container overhead
    audio_bps = 64_000
    video_bps = target_bits / video_s - audio_bps
    for attempt in range(4):
        kbps = int(video_bps / 1000)
        if kbps < 120:
            print(f"  warning: video is long ({video_s:.0f}s) for {limit_mb} MB; quality will be low")
            kbps = 120
        # Screen recordings are mostly static text, so they stay sharp at 720p on far less
        # bitrate than camera footage would need.
        height = 720 if kbps >= 450 else 540 if kbps >= 250 else 480
        common = ["-vf", f"scale=-2:{height}:flags=lanczos,format=yuv420p", "-c:v", "libx264",
                  "-preset", "slow", "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.5)}k",
                  "-bufsize", f"{kbps * 2}k", "-passlogfile", str(work / "x264pass")]
        run([ffmpeg, "-y", "-hide_banner", "-i", str(src), *common, "-pass", "1", "-an", "-f", "null", "-"],
            f"analyze for {out.name} (pass 1, {height}p @ {kbps} kbps)")
        run([ffmpeg, "-y", "-hide_banner", "-i", str(src), *common, "-pass", "2",
             "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-movflags", "+faststart", str(out)],
            f"encode {out.name} (pass 2)")
        size_mb = out.stat().st_size / 1_000_000
        if size_mb <= limit_mb:
            return size_mb
        video_bps *= 0.85 * limit_mb / size_mb
        print(f"  {size_mb:.1f} MB is over {limit_mb} MB, retrying with a lower bitrate")
    return out.stat().st_size / 1_000_000


# ---- 4. Subtitles ----------------------------------------------------------------------
def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_text(text: str, max_chars: int = 84) -> list[str]:
    """Split narration into subtitle-sized chunks at sentence or comma boundaries."""
    pieces = re.split(r"(?<=[.!?,;:])\s+", text.strip())
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def write_srt(plan: list[dict], scenes: list[dict], out: Path):
    cues = []
    for clip in plan:
        chunks = split_text(scenes[clip["index"]]["voice"])
        total = sum(len(c) for c in chunks)
        t = clip["start"]
        for chunk in chunks:
            d = clip["play"] * len(chunk) / total     # time shared by text length
            cues.append((t, t + d, chunk))
            t += d
    lines = [f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n" for n, (a, b, text) in enumerate(cues, 1)]
    out.write_text("\n".join(lines), encoding="utf-8")


# ---- main --------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="demo/scenes.json")
    ap.add_argument("--out", default="demo/output")
    ap.add_argument("--voice", default="en-US-AriaNeural", help="edge-tts voice name")
    ap.add_argument("--rate", default="+0%", help='speaking rate, e.g. "+10%%" or "-5%%"')
    ap.add_argument("--max-speed", type=float, default=1.35, help="max speed-up for a clip that is too long")
    ap.add_argument("--keep", type=float, default=0.6, help="seconds kept at each end of a cut wait")
    ap.add_argument("--min-cut", type=float, default=1.0, help="don't cut waits shorter than this (seconds)")
    ap.add_argument("--github-mb", type=float, default=9.5, help="size limit for demo_github.mp4")
    args = ap.parse_args()
    if not re.fullmatch(r"[+-]\d+%", args.rate):
        sys.exit('--rate must look like "+10%" or "-5%"')

    out = Path(args.out)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(Path(args.scenes).read_text(encoding="utf-8"))["scenes"]
    timings = json.loads((out / "timings.json").read_text(encoding="utf-8"))
    if timings.get("failed_scene"):
        sys.exit(f"The recording failed in scene '{timings['failed_scene']}'. Fix it and record again first.")
    raw = out / timings.get("video", "raw_recording.webm")
    vp = timings.get("viewport", {"width": 1920, "height": 1080})

    ffmpeg, ffprobe = find_ffmpeg()
    clips = synthesize(scenes, args.voice, args.rate, work)
    clip_durs = [duration(ffprobe, p) if p else 0.0 for p in clips]

    ranges = cut_ranges(timings, args.keep, args.min_cut)
    cut = work / "cut.mp4"
    cut_video(ffmpeg, raw, ranges, (vp["width"], vp["height"]), cut)
    video_s = duration(ffprobe, cut)

    plan = plan_clips(timings, ranges, clips, clip_durs, video_s, args.max_speed)
    youtube = out / "demo_youtube.mp4"
    mix_and_export(ffmpeg, cut, plan, video_s, youtube)
    github_mb = export_github(ffmpeg, youtube, video_s, out / "demo_github.mp4", args.github_mb, work)
    write_srt(plan, scenes, out / "subtitles.srt")

    removed = sum(b - a for a, b in ranges)
    print(f"\nDone. {video_s:.0f}s video ({removed:.0f}s of waiting cut out)")
    print(f"  {youtube}  ({youtube.stat().st_size / 1_000_000:.1f} MB)")
    print(f"  {out / 'demo_github.mp4'}  ({github_mb:.1f} MB)")
    print(f"  {out / 'subtitles.srt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
