# TAILOR: Trimming the Long-Tail of Visual World Modeling Evaluation

## Repository Layout

```text
.
+-- data_engine/
|   +-- action_ontology.json
|   +-- generate.py
|   +-- stage1_run_all.py
|   +-- stage_1_run_all_parallel.py
|   +-- stage2_run_all.py
|   +-- stage2_run_all_parallel.py
|   +-- s1_task_generation.py
|   +-- s2_creative_generation.py
|   +-- s3_adversarial_generation.py
|   +-- s4_revise_instances.py
|   +-- s5_prompt_generation.py
|   +-- s6_rubric_generation.py
|   +-- prompts/
+-- evaluation/
    +-- auto_eval/
    |   +-- run_generation_models.py
    |   +-- command.sh
    |   +-- utils/
    +-- human_eval/
        +-- app.py
```

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai gradio google-genai torch diffusers transformers
```

The repository does not currently include a pinned `requirements.txt`. Some generation backends may require additional model-specific packages, GPU support, or credentials.

Set API keys as needed:

```bash
export OPENAI_API_KEY="..."
export GOOGLE_GENAI_API_KEY="..."
```

`OPENAI_API_KEY` is used by `data_engine/generate.py` and the OpenAI auto-eval runner. `GOOGLE_GENAI_API_KEY` is used by the Google-backed `nanobanana` and `veo3` runners.

## Data Generation

The data engine is split into two stages.

### Stage 1: Task Construction

Stage 1 starts from an action ontology and creates task candidates, creative tool variants, and impossible-tool variants.

Sequential:

```bash
python data_engine/stage1_run_all.py
```

Parallel:

```bash
python data_engine/stage_1_run_all_parallel.py --jobs 10
```

Outputs are written under `tasks/`, including per-action files such as:

```text
tasks/<action_name>/s1_task.json
tasks/<action_name>/s2_creative.json
tasks/<action_name>/final_task.json
tasks/all_tasks.json
```

Note: the Stage 1 scripts currently look for `action_ontology.json` at the repository root. If you use the checked-in ontology at `data_engine/action_ontology.json`, copy or symlink it before running Stage 1:

```bash
cp data_engine/action_ontology.json action_ontology.json
```

### Stage 2: Prompt and Rubric Generation

Stage 2 expects a filtered task file:

```text
tasks/filtered_tasks.json
```

Sequential:

```bash
python data_engine/stage2_run_all.py
```

For a small test run:

```bash
python data_engine/stage2_run_all.py --limit 5
```

Parallel:

```bash
python data_engine/stage2_run_all_parallel.py --jobs 10
```

Stage 2 writes:

```text
tasks/revised_instances.json
tasks/selected_tasks_with_prompts.json
tasks/selected_tasks_with_rubrics.json
```

The parallel runner uses different default output names for the final prompt/rubric files:

```text
tasks/all_tasks_with_prompts.json
tasks/all_tasks_with_rubrics.json
```

You can override those paths:

```bash
python data_engine/stage2_run_all_parallel.py \
  --jobs 10 \
  --filtered tasks/filtered_tasks.json \
  --prompts tasks/selected_tasks_with_prompts.json \
  --rubrics tasks/selected_tasks_with_rubrics.json
```

## Automatic Media Generation

The auto-eval dispatcher runs one or more image/video generation backends over a prompt JSON file.

Dry run all commands:

```bash
python evaluation/auto_eval/run_generation_models.py \
  --models all \
  --json tasks/selected_tasks_with_prompts.json \
  --dry-run
```

Run all image models:

```bash
python evaluation/auto_eval/run_generation_models.py \
  --models image \
  --json tasks/selected_tasks_with_prompts.json \
  --results-dir evaluation/auto_eval/results \
  --skip-existing
```

Run all video models:

```bash
python evaluation/auto_eval/run_generation_models.py \
  --models video \
  --json tasks/selected_tasks_with_prompts.json \
  --results-dir evaluation/auto_eval/results \
  --skip-existing
```

Run selected models:

```bash
python evaluation/auto_eval/run_generation_models.py \
  --models qwen wan veo3 \
  --json tasks/selected_tasks_with_prompts.json \
  --results-dir evaluation/auto_eval/results \
  --skip-existing
```

Supported model keys:

```text
qwen, z-image, nanobanana, openai, wan, hunyuan, veo3
```

Aliases:

```text
all, image, images, video, videos
```

Useful options:

- `--limit N`: process a small subset for supported runners.
- `--seed N`: set the base seed for local diffusion runners.
- `--aspect 16:9`: set image aspect ratio for supported image runners.
- `--device cuda:0`: set GPU device for supported local runners.
- `--workers N`: set OpenAI worker count.
- `--api-key KEY`: pass a Google GenAI key for Google-backed runners.
- `--image-only` / `--video-only`: restrict OpenAI's mixed media runner.

Generated files are saved under per-model subdirectories in `evaluation/auto_eval/results/`.

## Human Annotation

The human annotation app is a Gradio UI for rating generated media against generated rubric items.

Expected inputs:

```text
evaluation/human_eval/rubrics.json
evaluation/results/<generated media files>
```

The app samples tasks deterministically per annotator UUID, shows image/video outputs, displays task context and rubric items, and saves annotations.

Run:

```bash
python evaluation/human_eval/app.py
```

Outputs:

```text
evaluation/human_eval/user_results/<uuid>/config.json
evaluation/human_eval/user_results/<uuid>/annotations.jsonl
```

Each annotation record includes the task, prompt, media path, rubric answers, issue flags, physical realism score, perceptual quality score, comments, and timestamp.

## File Naming for Human Eval Media

The human annotation app expects generated media filenames to match this pattern:

```text
task<task_id>_<slugified_task_goal>_inst<instance_index>_<tool_type>_<tool>_<prompt_label>.<ext>
```

Examples:

```text
task001_stack-blocks_inst000_original_tool_box_predictive_image.png
task001_stack-blocks_inst000_original_tool_box_predictive_video.mp4
```

Prompt labels are:

```text
predictive_image
descriptive_image
predictive_video
descriptive_video
```

Image prompts use `.png`; video prompts use `.mp4`.

## Typical End-to-End Flow

```bash
# 1. Prepare ontology for Stage 1 if needed.
cp data_engine/action_ontology.json action_ontology.json

# 2. Generate candidate tasks.
python data_engine/stage_1_run_all_parallel.py --jobs 10

# 3. Create or curate tasks/filtered_tasks.json.
# Human evaluation and filtering here.

# 4. Generate prompts and rubrics.
python data_engine/stage2_run_all_parallel.py \
  --jobs 10 \
  --filtered tasks/filtered_tasks.json \
  --prompts tasks/selected_tasks_with_prompts.json \
  --rubrics tasks/selected_tasks_with_rubrics.json

# 5. Generate media.
python evaluation/auto_eval/run_generation_models.py \
  --models all \
  --json tasks/selected_tasks_with_prompts.json \
  --results-dir evaluation/auto_eval/results \
  --skip-existing

# 6. Prepare human-eval inputs.
cp tasks/selected_tasks_with_rubrics.json evaluation/human_eval/rubrics.json

# 7. Run annotation UI.
python evaluation/human_eval/app.py
```

Depending on the generation backend, you may need to copy or normalize generated media into the filename and directory layout expected by `evaluation/human_eval/app.py`.
