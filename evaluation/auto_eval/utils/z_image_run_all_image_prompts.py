import argparse
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")

import torch
from diffusers import ZImagePipeline

# Default paths (relative to script dir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "selected_tasks_with_prompts.json")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "generated_images_z")

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
}
DEFAULT_ASPECT = "16:9"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Z-Image on all predictive/descriptive prompts from selected_tasks_with_prompts.json"
    )
    p.add_argument("--json", default=DEFAULT_JSON, help="Path to selected_tasks_with_prompts.json")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save generated images")
    p.add_argument(
        "--aspect",
        choices=list(ASPECT_RATIOS),
        default=DEFAULT_ASPECT,
        help="Aspect ratio for generated images",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip prompts whose output file already exists")
    p.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    p.add_argument("--device", default="cuda", help="Device (e.g. cuda or cuda:0)")
    return p.parse_args()


def load_pipeline(device="cuda"):
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    return pipe


def generate_image(pipe, prompt: str, width: int, height: int, seed: int, device: str):
    negative_prompt = ""
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        cfg_normalization=False,
        num_inference_steps=50,
        guidance_scale=4,
        generator=generator,
    ).images[0]
    return image


def main():
    args = parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    width, height = ASPECT_RATIOS[args.aspect]
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Z-Image pipeline...", flush=True)
    pipe = load_pipeline(args.device)

    total = 0
    skipped = 0
    for task in tasks:
        task_id = task["task_id"]
        instances = task.get("evaluation_instances", [])
        for inst_idx, inst in enumerate(instances):
            for prompt_type, key in [
                ("predictive", "predictive_image_prompt"),
                ("descriptive", "descriptive_image_prompt"),
            ]:
                prompt = inst.get(key)
                if not prompt or not prompt.strip():
                    continue
                seed = args.seed + task_id * 10000 + inst_idx * 10 + (1 if prompt_type == "descriptive" else 0)
                fname = f"task{task_id}_inst{inst_idx}_{prompt_type}.png"
                out_path = os.path.join(args.out_dir, fname)
                if args.skip_existing and os.path.isfile(out_path):
                    skipped += 1
                    continue
                total += 1
                print(f"[{total}] task_id={task_id} inst={inst_idx} {prompt_type} -> {fname}", flush=True)
                try:
                    image = generate_image(pipe, prompt, width, height, seed, args.device)
                    image.save(out_path)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)

    print(f"Done. Generated {total - skipped} new images, skipped {skipped} existing.", flush=True)


if __name__ == "__main__":
    main()
