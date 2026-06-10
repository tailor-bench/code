import argparse
import json
import os
import re
import sys

os.system("export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "selected_tasks_with_prompts.json")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "results", "wan_video")

HEIGHT = 720
WIDTH = 1280
NUM_FRAMES = 101
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 4.0
GUIDANCE_SCALE_2 = 3.0
FPS = 24
MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
NEGATIVE_PROMPT = ()


def slugify(text: str, max_len: int = 60) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "item"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Wan (Wan2.2-T2V) on all predictive/descriptive video prompts from selected_tasks_with_prompts.json"
    )
    p.add_argument("--json", default=DEFAULT_JSON, help="Path to selected_tasks_with_prompts.json")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save generated videos")
    p.add_argument("--skip-existing", action="store_true", help="Skip prompts whose output file already exists")
    p.add_argument("--seed", type=int, default=42, help="Base seed (per-task/instance/type derived)")
    p.add_argument("--limit", type=int, default=None, help="Max number of tasks to process (for testing)")
    return p.parse_args()


def load_pipeline(dtype=torch.bfloat16):
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=dtype)
    return pipe


def generate_video(pipe, prompt: str, out_path: str, seed: int) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE_SCALE,
        guidance_scale_2=GUIDANCE_SCALE_2,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
    ).frames[0]
    export_to_video(output, out_path, fps=FPS)


def main():
    args = parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if args.limit is not None:
        tasks = tasks[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Wan pipeline...", flush=True)
    pipe = load_pipeline()

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

                seed = args.seed + task_id * 10000 + inst_idx * 10 + (1 if "descriptive" in prompt_type else 0)
                fname = f"{base}_{prompt_type}.mp4"
                out_path = os.path.join(args.out_dir, fname)

                if args.skip_existing and os.path.isfile(out_path):
                    skipped += 1
                    continue

                total += 1
                print(f"[{total}] task_id={task_id} inst={inst_idx} {prompt_type} -> {fname}", flush=True)
                try:
                    generate_video(pipe, prompt, out_path, seed)
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
