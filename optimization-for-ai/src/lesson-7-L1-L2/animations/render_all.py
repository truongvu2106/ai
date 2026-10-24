"""Render all Lesson 7 Manim scenes locally.

Examples:
  python animations/render_all.py --quality low
  python animations/render_all.py --quality high --scene l2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE_FILE = ROOT / "animations" / "lesson7_scenes.py"
OUTPUT_DIR = ROOT / "outputs" / "manim_media"
SCENES = {
    "hook": "OverfittingHook",
    "generalization": "TrainValidationGap",
    "l2": "L2Regularization",
    "l1-l2": "L1VersusL2",
    "strength": "RegularizationStrengthAndRecap",
}
QUALITY = {"low": "l", "medium": "m", "high": "h", "production": "k"}
QUALITY_DIR = {
    "low": "480p15",
    "medium": "720p30",
    "high": "1080p60",
    "production": "1440p60",
}


def apply_speed(video_path: Path, speed: float) -> None:
    if speed == 1.0:
        return

    temp_path = video_path.with_name(f"{video_path.stem}.speed.tmp{video_path.suffix}")
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
        "-filter:v", f"setpts=PTS/{speed:g}",
        "-fps_mode", "passthrough",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_path),
    ]
    try:
        result = subprocess.run(command, check=False)
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg is required for --speed values other than 1.0.") from error
    if result.returncode != 0:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"FFmpeg failed to change video speed for {video_path}.")
    os.replace(temp_path, video_path)
    print(f"Applied {speed:g}x speed: {video_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Lesson 7 Manim animations.")
    parser.add_argument("--quality", choices=QUALITY, default="low")
    parser.add_argument("--scene", choices=["all", *SCENES], default="all")
    parser.add_argument(
        "--speed", type=float, default=1.0, metavar="MULTIPLIER",
        help="Playback speed (0.5 is half speed, 2 is double speed; default: 1).",
    )
    args = parser.parse_args()
    if not 0.25 <= args.speed <= 4.0:
        parser.error("--speed must be between 0.25 and 4.0.")
    selected = list(SCENES.items()) if args.scene == "all" else [(args.scene, SCENES[args.scene])]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for key, scene_name in selected:
        command = [
            sys.executable, "-m", "manim", f"-q{QUALITY[args.quality]}",
            "--media_dir", str(OUTPUT_DIR), str(SCENE_FILE), scene_name,
        ]
        print("\nRendering:", scene_name)
        print("Command:", " ".join(command))
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(f"Rendering {scene_name} failed with exit code {result.returncode}.")
            return result.returncode
        if args.speed != 1.0:
            video_path = (
                OUTPUT_DIR / "videos" / SCENE_FILE.stem
                / QUALITY_DIR[args.quality] / f"{scene_name}.mp4"
            )
            if not video_path.is_file():
                print(f"Rendered video was not found: {video_path}", file=sys.stderr)
                return 1
            try:
                apply_speed(video_path, args.speed)
            except RuntimeError as error:
                print(error, file=sys.stderr)
                return 1

    print(f"\nFinished. Check rendered files under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
