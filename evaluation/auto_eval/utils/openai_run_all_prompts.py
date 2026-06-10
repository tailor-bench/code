#!/usr/bin/env python3
from openai import OpenAI
import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Logging (thread-safe)
# ---------------------------------------------------------------------------

_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        print(msg, file=sys.stderr, flush=True)


def slugify(text, max_len=60):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "item"


def generate_image(client, prompt, out_path, skip_existing=True):
    if skip_existing and os.path.isfile(out_path):
        log(f"  Skip (exists): {out_path}")
        return True
    try:
        response = client.responses.create(
            model="gpt-5",
            input="Task: Image Generation\n" + prompt,
            tools=[{"type": "image_generation"}],
        )
        while response.status != "completed":
            time.sleep(1)
            response = client.responses.retrieve(response.id)
        image_data = []
        if response.output:
            for output in response.output:
                if getattr(output, "type", None) == "image_generation_call":
                    result = getattr(output, "result", None)
                    if result:
                        image_data.append(result)
        if not image_data:
            log(f"  No image in response for {out_path}")
            return False
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(image_data[0]))
        log(f"  Saved: {out_path}")
        return True
    except Exception as e:
        log(f"  Error {out_path}: {e}")
        return False


def generate_video(client, prompt, out_path, skip_existing=True):
    if skip_existing and os.path.isfile(out_path):
        log(f"  Skip (exists): {out_path}")
        return True
    try:
        video = client.videos.create(
            model="sora-2",
            size="1280x720",
            seconds="4",
            prompt=prompt,
        )
        while video.status in ("in_progress", "queued"):
            video = client.videos.retrieve(video.id)
            time.sleep(10)
        if video.status == "failed":
            msg = getattr(getattr(video, "error", None),
                          "message", "Video generation failed")
            log(f"  Failed {out_path}: {msg}")
            return False
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        content = client.videos.download_content(video.id, variant="video")
        content.write_to_file(out_path)
        log(f"  Saved: {out_path}")
        return True
    except Exception as e:
        log(f"  Error {out_path}: {e}")
        return False


def main():
    p = argparse.ArgumentParser(
        description="Generate images/videos for all prompts in selected_tasks_with_prompts.json")
    p.add_argument(
        "--json", default="selected_tasks_with_prompts.json", help="Path to tasks JSON")
    p.add_argument("--output-dir", default="openai_results",
                   help="Output directory for generated files")
    p.add_argument("--image-only", action="store_true",
                   help="Only generate image prompts")
    p.add_argument("--video-only", action="store_true",
                   help="Only generate video prompts")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Regenerate even if output file exists")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task indices (0-based), e.g. 0,2,4. Default: all")
    p.add_argument("--workers", type=int, default=10,
                   help="Number of parallel workers (default: 8)")
    p.add_argument("--delay", type=float, default=2.0,
                   help="Seconds to wait between API calls when workers=1 (rate limiting)")
    args = p.parse_args()

    do_image = not args.video_only
    do_video = not args.image_only

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json) as f:
        tasks = json.load(f)

    task_indices = range(len(tasks))
    if args.tasks is not None:
        task_indices = [int(x.strip()) for x in args.tasks.split(",")]

    skip = not args.no_skip_existing
    out_dir = args.output_dir.rstrip("/")

    jobs = []
    for ti in task_indices:
        if ti < 0 or ti >= len(tasks):
            continue
        task = tasks[ti]
        instances = task.get("evaluation_instances", [])
        for ii, inst in enumerate(instances):
            # Prefer explicit task_id from JSON if present; fall back to index.
            raw_task_id = task.get("task_id", ti)
            try:
                task_id = int(raw_task_id)
            except (TypeError, ValueError):
                task_id = ti

            task_goal = task.get("task_goal", f"task{task_id:03d}")
            tool_type = inst.get("tool_type", "unknown")
            tool_name = inst.get("tool", "tool")
            base = (
                f"{out_dir}/task{task_id:03d}_"
                f"{slugify(task_goal)}_"
                f"inst{ii:03d}_"
                f"{slugify(tool_type)}_"
                f"{slugify(tool_name)}"
            )
            if do_image:
                # JSON uses "predictive_image_prompt" and "descriptive_image_prompt"
                for key in ("predictive_image_prompt", "descriptive_image_prompt"):
                    label = key.replace("_image_prompt", "_image")
                    prompt = inst.get(key)
                    if prompt:
                        jobs.append(("image", prompt, f"{base}_{label}.png"))
            if do_video:
                # JSON uses "predictive_video_prompt" and "descriptive_video_prompt"
                for key in ("predictive_video_prompt", "descriptive_video_prompt"):
                    label = key.replace("_video_prompt", "_video")
                    prompt = inst.get(key)
                    if prompt:
                        jobs.append(("video", prompt, f"{base}_{label}.mp4"))

    log(f"Running {len(jobs)} jobs with {args.workers} workers")

    def run_one(job):
        kind, prompt, out_path = job
        client = OpenAI()
        # Try once, and on failure retry a single time.
        for attempt in range(2):
            if kind == "image":
                ok = generate_image(client, prompt, out_path, skip_existing=skip)
            else:
                ok = generate_video(client, prompt, out_path, skip_existing=skip)
            if ok:
                return True
            if attempt == 0:
                log(f"  Retry once: {out_path}")
                time.sleep(1)
        return False

    total_ok = 0
    total_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                if future.result():
                    total_ok += 1
                else:
                    total_fail += 1
            except Exception as e:
                total_fail += 1
                log(f"Worker error: {e}")

    log(f"Done. OK: {total_ok}, Failed: {total_fail}")
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
