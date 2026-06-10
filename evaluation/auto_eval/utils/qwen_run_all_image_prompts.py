import argparse
import json
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

os.system("export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")

import torch
from diffusers import DiffusionPipeline
import transformers.generation.configuration_utils as gen_config_utils

GenerationConfig = gen_config_utils.GenerationConfig
_orig_from_model_config = GenerationConfig.from_model_config


@classmethod
def _patched_from_model_config(cls, model_config):
    config_dict = model_config.to_dict()
    config_dict.pop("_from_model_config", None)
    config_dict = {k: v for k, v in config_dict.items() if v is not None}
    generation_config = cls.from_dict(
        config_dict, return_unused_kwargs=False, _from_model_config=True)
    decoder_config = model_config.get_text_config(decoder=True)
    if decoder_config is not model_config:
        default_generation_config = GenerationConfig()
        decoder_config_dict = decoder_config if isinstance(
            decoder_config, dict) else decoder_config.to_dict()
        for attr in generation_config.to_dict():
            is_unset = getattr(generation_config, attr) == getattr(
                default_generation_config, attr)
            if attr in decoder_config_dict and is_unset:
                setattr(generation_config, attr, decoder_config_dict[attr])
    if generation_config.return_dict_in_generate is False:
        if any(
            getattr(generation_config, f, False) for f in generation_config.extra_output_flags
        ):
            generation_config.return_dict_in_generate = True
    generation_config._original_object_hash = hash(generation_config)
    return generation_config


GenerationConfig.from_model_config = _patched_from_model_config


# Default paths (relative to script dir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "selected_tasks_with_prompts.json")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "generated_images")

# Generation defaults (from qwen-img.py)
POSITIVE_MAGIC = ", Ultra HD, 4K, cinematic composition."
NEGATIVE_PROMPT = " "
ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}
DEFAULT_ASPECT = "16:9"


def parse_args():
    p = argparse.ArgumentParser(description="Run Qwen-Image on all predictive/descriptive prompts from selected_tasks_with_prompts.json")
    p.add_argument("--json", default=DEFAULT_JSON, help="Path to selected_tasks_with_prompts.json")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save generated images")
    p.add_argument("--aspect", choices=list(ASPECT_RATIOS), default=DEFAULT_ASPECT, help="Aspect ratio for generated images")
    p.add_argument("--skip-existing", action="store_true", help="Skip prompts whose output file already exists")
    p.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility (per-task/instance/type derived)")
    return p.parse_args()


def load_pipeline():
    model_name = "Qwen/Qwen-Image-2512"
    torch_dtype = torch.bfloat16
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to("cuda")
    return pipe


def generate_image(pipe, prompt: str, width: int, height: int, seed: int):
    full_prompt = prompt + POSITIVE_MAGIC
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=generator,
    )
    return out.images[0]


def main():
    args = parse_args() 

    with open(args.json, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    width, height = ASPECT_RATIOS[args.aspect]
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Qwen-Image pipeline...", flush=True)
    pipe = load_pipeline()

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
                # Deterministic seed per (task_id, inst_idx, prompt_type)
                seed = args.seed + task_id * 10000 + inst_idx * 10 + (1 if prompt_type == "descriptive" else 0)
                fname = f"task{task_id}_inst{inst_idx}_{prompt_type}.png"
                out_path = os.path.join(args.out_dir, fname)
                if args.skip_existing and os.path.isfile(out_path):
                    skipped += 1
                    continue
                total += 1
                print(f"[{total}] task_id={task_id} inst={inst_idx} {prompt_type} -> {fname}", flush=True)
                try:
                    image = generate_image(pipe, prompt, width, height, seed)
                    image.save(out_path)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)

    print(f"Done. Generated {total - skipped} new images, skipped {skipped} existing.", flush=True)


if __name__ == "__main__":
    main()
