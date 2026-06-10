import argparse
import json
import os
import re
import sys
import time

from google import genai
from google.genai import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "selected_tasks_with_prompts.json")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "results", "veo3_video")

VEO_MODEL = "veo-3.1-fast-generate-preview"
DURATION_SECONDS = 4
POLL_INTERVAL_SECONDS = 10


def slugify(text: str, max_len: int = 60) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "item"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Veo 3 on all predictive/descriptive video prompts from selected_tasks_with_prompts.json"
    )
    p.add_argument("--json", default=DEFAULT_JSON, help="Path to selected_tasks_with_prompts.json")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save generated videos")
    p.add_argument("--skip-existing", action="store_true", help="Skip prompts whose output file already exists")
    p.add_argument("--limit", type=int, default=None, help="Max number of tasks to process (for testing)")
    p.add_argument("--api-key", default=os.environ.get("GOOGLE_GENAI_API_KEY"), help="Google GenAI API key (or set GOOGLE_GENAI_API_KEY)")
    return p.parse_args()


def generate_video(client, prompt: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    operation = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            duration_seconds=DURATION_SECONDS,
        ),
    )
    while not operation.done:
        print("  Waiting for video generation...", flush=True)
        time.sleep(POLL_INTERVAL_SECONDS)
        operation = client.operations.get(operation)
    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)
    generated_video.video.save(out_path)


def main():
    args = parse_args()
    if not args.api_key:
        print("Error: Set GOOGLE_GENAI_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=args.api_key)

    with open(args.json, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if args.limit is not None:
        tasks = tasks[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    skipped = 0
    failed = 0

    for task_idx, task in enumerate(tasks):
        raw_task_id = task.get("task_id", task_idx)
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError):
            task_id = task_idx

        task_goal = task.get("task_goal", f"task{task_id:03d}")
        instances = task.get("evaluation_instances", [])

        for inst_idx, inst in enumerate(instances):
            tool_type = inst.get("tool_type", "unknown")
            tool_name = inst.get("tool", "tool")
            base = (
                f"task{task_id:03d}_"
                f"{slugify(task_goal)}_"
                f"inst{inst_idx:03d}_"
                f"{slugify(tool_type)}_"
                f"{slugify(tool_name)}"
            )

            for prompt_type, key in [
                ("predictive_video", "predictive_video_prompt"),
                ("descriptive_video", "descriptive_video_prompt"),
            ]:
                prompt = inst.get(key)
                if not prompt or not prompt.strip():
                    continue

                fname = f"{base}_{prompt_type}.mp4"
                out_path = os.path.join(args.out_dir, fname)

                if args.skip_existing and os.path.isfile(out_path):
                    skipped += 1
                    continue

                total += 1
                print(f"[{total}] task_id={task_id} inst={inst_idx} {prompt_type} -> {fname}", flush=True)
                try:
                    generate_video(client, prompt, out_path)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    failed += 1

    print(
        f"Done. Generated {total - skipped - failed} new videos, skipped {skipped} existing, failed {failed}.",
        flush=True,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
