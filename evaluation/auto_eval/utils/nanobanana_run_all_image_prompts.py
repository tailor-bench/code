import argparse
import json
import os
import sys

from google import genai

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "selected_tasks_with_prompts.json")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "results", "nanobanana_image")
MODEL_ID = "gemini-3.1-flash-image-preview"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Gemini (nanobanana) on all predictive/descriptive image prompts from selected_tasks_with_prompts.json"
    )
    p.add_argument("--json", default=DEFAULT_JSON, help="Path to selected_tasks_with_prompts.json")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save generated images")
    p.add_argument("--skip-existing", action="store_true", help="Skip prompts whose output file already exists")
    p.add_argument("--limit", type=int, default=None, help="Max number of tasks to process (for testing)")
    p.add_argument("--api-key", default=os.environ.get("GOOGLE_GENAI_API_KEY"), help="Google GenAI API key (or set GOOGLE_GENAI_API_KEY)")
    return p.parse_args()


def generate_image(client, prompt: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt],
    )
    for part in response.parts:
        if part.inline_data is not None:
            image = part.as_image()
            image.save(out_path)
            return
    raise RuntimeError("No image in response")


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

        instances = task.get("evaluation_instances", [])

        for inst_idx, inst in enumerate(instances):
            for prompt_type, key in [
                ("predictive", "predictive_image_prompt"),
                ("descriptive", "descriptive_image_prompt"),
            ]:
                prompt = inst.get(key)
                if not prompt or not prompt.strip():
                    continue

                fname = f"task{task_id}_inst{inst_idx}_{prompt_type}.png"
                out_path = os.path.join(args.out_dir, fname)

                if args.skip_existing and os.path.isfile(out_path):
                    skipped += 1
                    continue

                total += 1
                print(f"[{total}] task_id={task_id} inst={inst_idx} {prompt_type} -> {fname}", flush=True)
                try:
                    generate_image(client, prompt, out_path)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    failed += 1

    print(
        f"Done. Generated {total - skipped - failed} new images, skipped {skipped} existing, failed {failed}.",
        flush=True,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
