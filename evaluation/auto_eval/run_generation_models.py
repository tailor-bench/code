#!/usr/bin/env python3
"""Dispatch selected auto-eval generation prompts to one or more model runners."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


AUTO_EVAL_DIR = Path(__file__).resolve().parent
UTILS_DIR = AUTO_EVAL_DIR / "utils"
DEFAULT_JSON = UTILS_DIR / "selected_tasks_with_prompts.json"
DEFAULT_RESULTS_DIR = AUTO_EVAL_DIR / "results"


@dataclass(frozen=True)
class ModelRunner:
    name: str
    media: str
    script: str
    default_out_subdir: str
    out_flag: str = "--out-dir"
    supports_limit: bool = True
    supports_seed: bool = False
    supports_aspect: bool = False
    supports_device: bool = False
    supports_workers: bool = False
    supports_delay: bool = False
    supports_api_key: bool = False
    skip_existing_flag: str = "--skip-existing"


RUNNERS = {
    "qwen": ModelRunner(
        name="qwen",
        media="image",
        script="qwen_run_all_image_prompts.py",
        default_out_subdir="qwen_image",
        supports_seed=True,
        supports_aspect=True,
    ),
    "z-image": ModelRunner(
        name="z-image",
        media="image",
        script="z_image_run_all_image_prompts.py",
        default_out_subdir="z_image",
        supports_seed=True,
        supports_aspect=True,
        supports_device=True,
    ),
    "nanobanana": ModelRunner(
        name="nanobanana",
        media="image",
        script="nanobanana_run_all_image_prompts.py",
        default_out_subdir="nanobanana_image",
        supports_api_key=True,
    ),
    "openai": ModelRunner(
        name="openai",
        media="both",
        script="openai_run_all_prompts.py",
        default_out_subdir="openai",
        out_flag="--output-dir",
        supports_workers=True,
        supports_delay=True,
        supports_api_key=False,
        supports_limit=False,
        skip_existing_flag="",
    ),
    "wan": ModelRunner(
        name="wan",
        media="video",
        script="wan_run_all_video_prompts.py",
        default_out_subdir="wan_video",
        supports_seed=True,
    ),
    "hunyuan": ModelRunner(
        name="hunyuan",
        media="video",
        script="hunyuan_run_all_video_prompts.py",
        default_out_subdir="hunyuan_video",
        supports_seed=True,
        supports_device=True,
    ),
    "veo3": ModelRunner(
        name="veo3",
        media="video",
        script="veo3_run_all_video_prompts.py",
        default_out_subdir="veo3_video",
        supports_api_key=True,
    ),
}

ALIASES = {
    "all": tuple(RUNNERS),
    "images": tuple(name for name, runner in RUNNERS.items() if runner.media in ("image", "both")),
    "image": tuple(name for name, runner in RUNNERS.items() if runner.media in ("image", "both")),
    "videos": tuple(name for name, runner in RUNNERS.items() if runner.media in ("video", "both")),
    "video": tuple(name for name, runner in RUNNERS.items() if runner.media in ("video", "both")),
}


def parse_csv(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def expand_models(values: list[str]) -> list[str]:
    selected: list[str] = []
    for value in parse_csv(values):
        key = value.lower()
        names = ALIASES.get(key, (key,))
        for name in names:
            if name not in RUNNERS:
                valid = ", ".join(sorted([*RUNNERS, *ALIASES]))
                raise SystemExit(f"Unknown model '{value}'. Valid choices: {valid}")
            if name not in selected:
                selected.append(name)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more auto-eval image/video generation backends. "
            "This dispatches to the model-specific scripts in this directory."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Model names or aliases. Models: qwen, z-image, nanobanana, openai, "
            "wan, hunyuan, veo3. Aliases: all, image/images, video/videos."
        ),
    )
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="Path to selected_tasks_with_prompts.json")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Parent directory for per-model outputs")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch runner scripts")
    parser.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist when supported")
    parser.add_argument("--limit", type=int, default=None, help="Max number of tasks for runners that support --limit")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for local diffusion runners")
    parser.add_argument("--aspect", default=None, help="Aspect ratio for image runners that support it, e.g. 16:9")
    parser.add_argument("--device", default=None, help="Device for runners that expose --device, e.g. cuda:0")
    parser.add_argument("--workers", type=int, default=None, help="OpenAI worker count")
    parser.add_argument("--delay", type=float, default=None, help="OpenAI delay between calls when workers=1")
    parser.add_argument("--api-key", default=None, help="Google GenAI API key for Google-backed runners")
    parser.add_argument("--image-only", action="store_true", help="Only image prompts for runners that support both media")
    parser.add_argument("--video-only", action="store_true", help="Only video prompts for runners that support both media")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser.parse_args()


def add_if_supported(cmd: list[str], runner: ModelRunner, flag: str, value, supported: bool) -> None:
    if value is not None and supported:
        cmd.extend([flag, str(value)])


def command_for_runner(args: argparse.Namespace, runner: ModelRunner) -> list[str]:
    script = UTILS_DIR / runner.script
    out_dir = Path(args.results_dir).expanduser() / runner.default_out_subdir

    cmd = [
        args.python,
        str(script),
        "--json",
        str(Path(args.json).expanduser()),
        runner.out_flag,
        str(out_dir),
    ]

    if args.skip_existing and runner.skip_existing_flag:
        cmd.append(runner.skip_existing_flag)
    if args.skip_existing and runner.name == "openai":
        pass
    elif not args.skip_existing and runner.name == "openai":
        cmd.append("--no-skip-existing")

    add_if_supported(cmd, runner, "--limit", args.limit, runner.supports_limit)
    if runner.name == "openai" and args.limit is not None:
        cmd.extend(["--tasks", ",".join(str(i) for i in range(args.limit))])
    add_if_supported(cmd, runner, "--seed", args.seed, runner.supports_seed)
    add_if_supported(cmd, runner, "--aspect", args.aspect, runner.supports_aspect)
    add_if_supported(cmd, runner, "--device", args.device, runner.supports_device)
    add_if_supported(cmd, runner, "--workers", args.workers, runner.supports_workers)
    add_if_supported(cmd, runner, "--delay", args.delay, runner.supports_delay)
    add_if_supported(cmd, runner, "--api-key", args.api_key, runner.supports_api_key)

    if runner.name == "openai":
        if args.image_only and args.video_only:
            raise SystemExit("Use only one of --image-only or --video-only")
        if args.image_only:
            cmd.append("--image-only")
        if args.video_only:
            cmd.append("--video-only")

    return cmd


def main() -> int:
    args = parse_args()
    selected = expand_models(args.models)

    json_path = Path(args.json).expanduser()
    if not json_path.is_file():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        return 2

    failed: list[str] = []
    for name in selected:
        runner = RUNNERS[name]
        cmd = command_for_runner(args, runner)
        printable = " ".join(subprocess.list2cmdline([part]) for part in cmd)
        print(f"\n==> {name}: {printable}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=AUTO_EVAL_DIR)
        if result.returncode != 0:
            failed.append(name)

    if failed:
        print(f"\nFailed runners: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
