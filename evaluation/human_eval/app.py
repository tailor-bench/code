import gradio as gr
import json
import os
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

_GRADIO_TMP = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "gradio_tmp")
os.makedirs(_GRADIO_TMP, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", _GRADIO_TMP)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RUBRICS_PATH = os.path.join(BASE_DIR, "rubrics.json")
RESULTS_ROOT = os.path.join(BASE_DIR, "user_results")
MEDIA_ROOT = os.path.join(PROJECT_ROOT, "evaluation", "results")


def slugify(text: str, max_len: int = 60) -> str:
    text = (text or "").lower()
    text = __import__("re").sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "item"


PROMPT_TO_LABEL = {
    "predictive_image_prompt": "predictive_image",
    "descriptive_image_prompt": "descriptive_image",
    "predictive_video_prompt": "predictive_video",
    "descriptive_video_prompt": "descriptive_video",
}

PROMPT_TO_MEDIA_TYPE = {
    "predictive_image_prompt": "image",
    "descriptive_image_prompt": "image",
    "predictive_video_prompt": "video",
    "descriptive_video_prompt": "video",
}

PROMPT_TO_RUBRIC_KEY = {
    "predictive_image_prompt": "predictive_image_rubric",
    "descriptive_image_prompt": "descriptive_image_rubric",
    "predictive_video_prompt": "predictive_video_rubric",
    "descriptive_video_prompt": "descriptive_video_rubric",
}

# Upper bound on number of rubric checklist items per instance
MAX_RUBRIC_ITEMS = 60


@dataclass
class FlatInstance:
    uuid: str
    instance_index: int
    task_index: int
    eval_index: int
    task_id: int
    task_goal: str
    tool_type: str
    tool: str
    expected_outcome: str
    prompt_key: str
    prompt_label: str
    prompt_text: str
    media_type: str  # "image" or "video"
    media_path: str
    rubric_key: str


def load_tasks() -> List[dict]:
    with open(RUBRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_media_path(task: dict, task_idx: int, inst_idx: int, inst: dict, prompt_key: str) -> str:
    task_id = task.get("task_id", task_idx)
    task_goal = task.get("task_goal", f"task{task_id:03d}")
    tool_type = inst.get("tool_type", "unknown")
    tool_name = inst.get("tool", "tool")

    base_name = (
        f"task{int(task_id):03d}_"
        f"{slugify(task_goal)}_"
        f"inst{inst_idx:03d}_"
        f"{slugify(tool_type)}_"
        f"{slugify(tool_name)}"
    )

    label = PROMPT_TO_LABEL[prompt_key]
    ext = "png" if PROMPT_TO_MEDIA_TYPE[prompt_key] == "image" else "mp4"
    filename = f"{base_name}_{label}.{ext}"
    return os.path.join(MEDIA_ROOT, filename)


def resolve_expected_outcome(task: dict, inst: dict) -> str:
    tool_type = inst.get("tool_type", "")
    if tool_type == "impossible":
        mapping = task.get("expected_outcome_impossible_tool", {}) or {}
        tool_name = inst.get("tool")
        if tool_name in mapping:
            return mapping[tool_name]
    return inst.get("expected_outcome") or task.get("expected_outcome", "")


def sample_task_indices_for_uuid(tasks: List[dict], user_uuid: str, k: int = 8) -> List[int]:
    n = len(tasks)
    if n <= k:
        return list(range(n))
    rnd = random.Random(user_uuid)
    return sorted(rnd.sample(range(n), k))


def expand_instances_for_uuid(tasks: List[dict], user_uuid: str) -> List[FlatInstance]:
    indices = sample_task_indices_for_uuid(tasks, user_uuid, k=8)
    flat_instances: List[FlatInstance] = []
    counter = 0

    for task_idx in indices:
        task = tasks[task_idx]
        eval_instances = task.get("evaluation_instances", []) or []
        for inst_idx, inst in enumerate(eval_instances):
            for prompt_key, label in PROMPT_TO_LABEL.items():
                prompt_text = inst.get(prompt_key)
                if not prompt_text:
                    continue
                media_type = PROMPT_TO_MEDIA_TYPE[prompt_key]
                rubric_key = PROMPT_TO_RUBRIC_KEY.get(prompt_key, "")
                media_path = build_media_path(
                    task, task_idx, inst_idx, inst, prompt_key
                )
                # Skip instances where the expected media file does not exist.
                if not os.path.isfile(media_path):
                    continue
                expected_outcome = resolve_expected_outcome(task, inst)

                flat_instances.append(
                    FlatInstance(
                        uuid=user_uuid,
                        instance_index=counter,
                        task_index=task_idx,
                        eval_index=inst_idx,
                        task_id=int(task.get("task_id", task_idx)),
                        task_goal=task.get("task_goal", ""),
                        tool_type=inst.get("tool_type", ""),
                        tool=inst.get("tool", ""),
                        expected_outcome=expected_outcome,
                        prompt_key=prompt_key,
                        prompt_label=label,
                        prompt_text=prompt_text,
                        media_type=media_type,
                        media_path=media_path,
                        rubric_key=rubric_key,
                    )
                )
                counter += 1

    return flat_instances


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_ROOT, exist_ok=True)


def get_user_dir(user_uuid: str) -> str:
    ensure_results_dir()
    d = os.path.join(RESULTS_ROOT, user_uuid)
    os.makedirs(d, exist_ok=True)
    return d


def load_or_create_config(tasks: List[dict], user_uuid: str) -> dict:
    user_dir = get_user_dir(user_uuid)
    cfg_path = os.path.join(user_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    indices = sample_task_indices_for_uuid(tasks, user_uuid, k=8)
    cfg = {
        "uuid": user_uuid,
        "task_indices": indices,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def expand_instances_from_config(tasks: List[dict], user_uuid: str, cfg: dict) -> List[FlatInstance]:
    indices = cfg.get("task_indices") or sample_task_indices_for_uuid(
        tasks, user_uuid, k=8)
    flat_instances: List[FlatInstance] = []
    counter = 0
    for task_idx in indices:
        task = tasks[task_idx]
        eval_instances = task.get("evaluation_instances", []) or []
        for inst_idx, inst in enumerate(eval_instances):
            for prompt_key, label in PROMPT_TO_LABEL.items():
                prompt_text = inst.get(prompt_key)
                if not prompt_text:
                    continue
                media_type = PROMPT_TO_MEDIA_TYPE[prompt_key]
                rubric_key = PROMPT_TO_RUBRIC_KEY.get(prompt_key, "")
                media_path = build_media_path(
                    task, task_idx, inst_idx, inst, prompt_key
                )
                # Skip instances where the expected media file does not exist.
                if not os.path.isfile(media_path):
                    continue
                expected_outcome = resolve_expected_outcome(task, inst)

                flat_instances.append(
                    FlatInstance(
                        uuid=user_uuid,
                        instance_index=counter,
                        task_index=task_idx,
                        eval_index=inst_idx,
                        task_id=int(task.get("task_id", task_idx)),
                        task_goal=task.get("task_goal", ""),
                        tool_type=inst.get("tool_type", ""),
                        tool=inst.get("tool", ""),
                        expected_outcome=expected_outcome,
                        prompt_key=prompt_key,
                        prompt_label=label,
                        prompt_text=prompt_text,
                        media_type=media_type,
                        media_path=media_path,
                        rubric_key=rubric_key,
                    )
                )
                counter += 1
    return flat_instances


# Global in-memory cache of instances per UUID.
TASKS_CACHE: Optional[List[dict]] = None
INSTANCES_CACHE: Dict[str, List[FlatInstance]] = {}


def get_tasks() -> List[dict]:
    global TASKS_CACHE
    if TASKS_CACHE is None:
        TASKS_CACHE = load_tasks()
    return TASKS_CACHE


def get_instances_for_uuid(user_uuid: str) -> List[FlatInstance]:
    tasks = get_tasks()
    if user_uuid not in INSTANCES_CACHE:
        cfg = load_or_create_config(tasks, user_uuid)
        INSTANCES_CACHE[user_uuid] = expand_instances_from_config(
            tasks, user_uuid, cfg)
    return INSTANCES_CACHE[user_uuid]


def flatten_rubric(task: dict, inst: dict, prompt_key: str) -> List[Dict[str, str]]:
    rubric_key = PROMPT_TO_RUBRIC_KEY.get(prompt_key, "")
    rubric = inst.get(rubric_key) or {}
    items: List[Dict[str, str]] = []
    for top_name, top_val in rubric.items():
        if not isinstance(top_val, dict):
            continue
        for sub_name, sub_val in top_val.items():
            if not isinstance(sub_val, dict):
                continue
            checklist = sub_val.get("checklist_items") or []
            for item in checklist:
                full_id = f"{top_name}.{sub_name}.{item['id']}"
                items.append(
                    {
                        "full_id": full_id,
                        "id": item["id"],
                        "category": top_name,
                        "sub_category": sub_name,
                        "question": item["question"],
                    }
                )
    return items


def format_rubric_choice_label(item: Dict[str, str]) -> str:
    def _pretty(name: str) -> str:
        name = (name or "").replace("_", " ").replace("-", " ").strip()
        if not name:
            return ""
        return name.title()

    category = _pretty(item.get("category") or "")
    sub_category = _pretty(item.get("sub_category") or "")
    parts = [p for p in (category, sub_category) if p]
    if parts:
        subtitle = " / ".join(parts)
        # Show category as a bold label line above the question.
        return f"**[{subtitle}]**\n{item['question']}"
    return item["question"]


def parse_checked_questions(labels: List[str]) -> List[str]:
    return list(labels or [])


def build_context_markdown(instance: FlatInstance) -> str:
    lines = [
        f"**Goal**: {instance.task_goal}\n",
        f"**Tool type**: {instance.tool_type}\n",
        f"**Tool**: {instance.tool}\n",
        f"**Expected outcome**: {instance.expected_outcome}\n",
        f"**Prompt({instance.prompt_label.replace('_', ' ').title()})**:",
        f"> {instance.prompt_text}",
    ]
    return "\n".join(lines)


def build_rubric_table(
    tasks: List[dict], instance: FlatInstance, rec: Optional[dict]
) -> Tuple[List[List[str]], List[Dict[str, str]]]:
    task = tasks[instance.task_index]
    eval_instances = task.get("evaluation_instances", [])
    inst: dict = {}
    if 0 <= instance.eval_index < len(eval_instances):
        inst = eval_instances[instance.eval_index]
    flat_items = flatten_rubric(task, inst, instance.prompt_key)
    rubric_answers = (rec or {}).get("rubric_answers") or {}
    issue_ids = set((rec or {}).get("rubric_issue_flags") or [])
    rows: List[List[str]] = []
    for item in flat_items:
        label = format_rubric_choice_label(item)
        full_id = item["full_id"]
        # Issue takes precedence over Yes when pre-filling.
        if full_id in issue_ids:
            answer = "Issue"
        elif rubric_answers.get(full_id):
            answer = "Yes"
        else:
            answer = "No"
        rows.append([label, answer])

    return rows, flat_items


def load_existing_annotations(user_uuid: str) -> Dict[int, dict]:
    user_dir = get_user_dir(user_uuid)
    path = os.path.join(user_dir, "annotations.jsonl")
    if not os.path.isfile(path):
        return {}
    by_index: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("uuid") != user_uuid:
                continue
            idx = rec.get("instance_index")
            if isinstance(idx, int):
                by_index[idx] = rec
    return by_index


def next_unannotated_index(user_uuid: str, instances: List[FlatInstance]) -> int:
    existing = load_existing_annotations(user_uuid)
    if not existing:
        return 0
    annotated = sorted(existing.keys())
    return min(len(annotated), len(instances) - 1)


def start_or_load_session(existing_uuid: str) -> Tuple[str, str, dict, List[List[str]], Optional[int], Optional[int], str]:
    tasks = get_tasks()
    if existing_uuid and existing_uuid.strip():
        user_uuid = existing_uuid.strip()
    else:
        user_uuid = str(uuid.uuid4())

    instances = get_instances_for_uuid(user_uuid)
    ann_by_index = load_existing_annotations(user_uuid)
    current_idx = next_unannotated_index(user_uuid, instances)
    instance = instances[current_idx]

    rec = ann_by_index.get(current_idx, {})
    context_md = build_context_markdown(instance)
    tasks_for_rubric = get_tasks()
    rubric_rows, _flat_items = build_rubric_table(
        tasks_for_rubric, instance, rec
    )

    # Pre-fill scores and comments if already annotated.
    physical = rec.get("physical_realism")
    perceptual = rec.get("perceptual_quality")
    comments = rec.get("comments", "")

    total = len(instances)
    completed = len(ann_by_index)
    progress = f"**Progress:** Instance {current_idx + 1} of {total}  |  Completed: {completed}"

    state = {
        "uuid": user_uuid,
        "current_index": current_idx,
    }

    return (
        user_uuid,
        progress,
        state,
        rubric_rows,
        physical,
        perceptual,
        comments,
    )


def render_instance(state: dict) -> Tuple[str, str, List[List[str]], Optional[int], Optional[int], str]:
    user_uuid = state.get("uuid")
    current_idx = state.get("current_index", 0)
    instances = get_instances_for_uuid(user_uuid)
    tasks = get_tasks()

    if current_idx < 0:
        current_idx = 0
    if current_idx >= len(instances):
        current_idx = len(instances) - 1
        state["current_index"] = current_idx

    instance = instances[current_idx]
    context_md = build_context_markdown(instance)

    # Media: we return the file path and indicate type via state; Gradio callbacks decide which to show.
    media_path = instance.media_path

    ann_by_index = load_existing_annotations(user_uuid)
    rec = ann_by_index.get(current_idx, {})
    rubric_rows, _flat_items = build_rubric_table(
        tasks, instance, rec
    )
    physical = rec.get("physical_realism")
    perceptual = rec.get("perceptual_quality")

    total = len(instances)
    completed = len(ann_by_index)
    progress = f"**Progress:** Instance {current_idx + 1} of {total}  |  Completed: {completed}"

    # For media type, we inject into state to be used in UI.
    state["media_type"] = instance.media_type
    state["media_path"] = media_path
    return (
        context_md,
        media_path,
        rubric_rows,
        physical,
        perceptual,
        progress,
    )


def go_previous(state: dict):
    if not state:
        return state
    state["current_index"] = max(0, state.get("current_index", 0) - 1)
    return state


def go_next(state: dict, instances: List[FlatInstance]) -> dict:
    if not state:
        return state
    idx = state.get("current_index", 0)
    if idx < len(instances) - 1:
        state["current_index"] = idx + 1
    return state


def save_and_advance(
    state: dict,
    rubric_table: List[List[str]],
    physical_realism: Optional[float],
    perceptual_quality: Optional[float],
    comments: str,
) -> dict:
    user_uuid = state.get("uuid")
    if not user_uuid:
        return state
    instances = get_instances_for_uuid(user_uuid)
    idx = state.get("current_index", 0)
    if idx < 0 or idx >= len(instances):
        return state

    instance = instances[idx]
    tasks = get_tasks()
    task = tasks[instance.task_index]
    eval_instances = task.get("evaluation_instances", []) or []
    inst_obj: dict = {}
    if 0 <= instance.eval_index < len(eval_instances):
        inst_obj = eval_instances[instance.eval_index]

    flat_items = flatten_rubric(task, inst_obj, instance.prompt_key)
    rubric_answers: Dict[str, bool] = {}
    issue_full_ids: set[str] = set()
    rows = rubric_table or []
    for idx_item, item in enumerate(flat_items):
        full_id = item["full_id"]
        answer = "No"
        if idx_item < len(rows):
            row = rows[idx_item] or []
            if len(row) >= 2:
                raw = str(row[1] or "").strip().lower()
                if raw == "yes":
                    answer = "Yes"
                elif raw == "issue":
                    answer = "Issue"
                elif raw == "no":
                    answer = "No"
        if answer == "Issue":
            rubric_answers[full_id] = True
            issue_full_ids.add(full_id)
        elif answer == "Yes":
            rubric_answers[full_id] = True
        else:
            rubric_answers[full_id] = False

    user_dir = get_user_dir(user_uuid)
    path = os.path.join(user_dir, "annotations.jsonl")
    record = {
        "uuid": user_uuid,
        "instance_index": idx,
        "task_index": instance.task_index,
        "task_id": instance.task_id,
        "tool_type": instance.tool_type,
        "tool": instance.tool,
        "prompt_key": instance.prompt_key,
        "prompt_label": instance.prompt_label,
        "media_type": instance.media_type,
        "media_path": instance.media_path,
        "prompt_text": instance.prompt_text,
        "expected_outcome": instance.expected_outcome,
        "rubric_key": instance.rubric_key,
        "rubric_answers": rubric_answers,
        "rubric_issue_flags": sorted(issue_full_ids),
        "physical_realism": physical_realism,
        "perceptual_quality": perceptual_quality,
        "comments": comments,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Move to next instance if possible.
    if idx < len(instances) - 1:
        state["current_index"] = idx + 1
    return state


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

    css = """
    :root {
      color-scheme: light !important;
    }
    html, body {
      background-color: #f5f5f7 !important;
      color: #111827 !important;
    }
    html.dark, body.dark, .dark {
      background-color: #f5f5f7 !important;
      color: #111827 !important;
    }

    /* Global readability */
    .gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
    .prose h2 { margin-bottom: 0.25rem; }
    .muted { color: rgba(60,60,60,0.75); font-size: 0.95rem; }

    /* Panels / cards */
    .card {
      border: 1px solid rgba(120,120,120,0.18);
      border-radius: 14px;
      padding: 14px 14px;
      background: rgba(255,255,255,0.78);
    }
    .tight > * { margin-top: 6px !important; }

    /* Scroll rubric area */
    .rubric-scroll {
      max-height: 62vh;
      overflow: auto;
      padding-right: 6px;
    }
    .rubric-item {
      border: 1px solid rgba(120,120,120,0.14);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.55);
      margin-bottom: 10px;
    }
    .rubric-q { font-size: 0.98rem; line-height: 1.25rem; }
    .rubric-sub { color: rgba(60,60,60,0.7); font-size: 0.85rem; margin-top: 3px; }

    /* Make radios more compact */
    .compact-radio label { margin-right: 10px !important; }

    /* Media box */
    .media-card { padding: 10px 12px; }

    /* Top bar */
    .topbar { padding: 12px 14px; }
    """

    with gr.Blocks(theme=theme, title="TAILOR Human Annotation", css=css) as demo:
        session_state = gr.State({})

        # ---------- Top bar ----------
        with gr.Row(elem_classes=["card", "topbar"]):
            with gr.Column(scale=6, elem_classes=["tight"]):
                gr.Markdown("## TAILOR Human Annotation",
                            elem_classes=["prose"])
                gr.Markdown(
                    "### Make sure to answer MCQ and Likert Questions for each instance. Add additional comments if needed.",
                    elem_classes=["muted"],
                )
                progress_md = gr.Markdown(value="**Progress:** —")
            with gr.Column(scale=4):
                with gr.Row():
                    existing_uuid = gr.Textbox(
                        label="Enter ID to start",
                        placeholder="Your ID",
                        scale=3,
                    )
                    start_btn = gr.Button(
                        "Start / Load", variant="primary", scale=1)

        # ---------- Main area ----------
        with gr.Row(equal_height=True):
            # LEFT: context + media
            with gr.Column(scale=6):
                with gr.Column(elem_classes=["card", "media-card", "tight"]):
                    context_md = gr.Markdown(value="")

                with gr.Column(elem_classes=["card", "media-card", "tight"]):
                    gr.Markdown("### Sample", elem_classes=["prose"])
                    media_image = gr.Image(
                        label="Generated Image", visible=False, interactive=False)
                    media_video = gr.Video(
                        label="Generated Video", visible=False, interactive=False)

            # RIGHT: tabs for rubric/scores/comments
            with gr.Column(scale=5, min_width=420):
                with gr.Tabs():
                    with gr.Tab("MCQ"):
                        gr.Markdown(
                            "Mark each item as **Yes**, **No**, or **Issue** (use *Issue* when it’s ambiguous or broken).",
                            elem_classes=["muted"],
                        )
                        with gr.Column(elem_classes=["rubric-scroll"]):
                            rubric_rows_components = []
                            for _ in range(MAX_RUBRIC_ITEMS):
                                with gr.Column(visible=False, elem_classes=["rubric-item"]) as item_box:
                                    q_md = gr.Markdown(
                                        value="", elem_classes=["rubric-q"])
                                    a_radio = gr.Radio(
                                        choices=["Yes", "No", "Issue"],
                                        value="No",
                                        show_label=False,
                                        elem_classes=["compact-radio"],
                                    )
                                rubric_rows_components.append(
                                    (item_box, q_md, a_radio))

                    with gr.Tab("Likert Questions"):
                        gr.Markdown(
                            "Use integer scores (0–5). If unsure, use **3** as a neutral default.",
                            elem_classes=["muted"],
                        )
                        physical_slider = gr.Slider(
                            minimum=0, maximum=5, step=1,
                            label="Physical Realism (0–5)",
                            value=3,
                        )
                        perceptual_slider = gr.Slider(
                            minimum=0, maximum=5, step=1,
                            label="Perceptual Quality (0–5)",
                            value=3,
                        )
                        with gr.Accordion("Score definitions", open=False):
                            gr.Markdown(
                                "**Physical Realism (0–5)**\n"
                                "- **0–1:** Clearly impossible or severely broken physics (e.g., objects float, intersect, or ignore gravity/contact).\n"
                                "- **2:** Multiple noticeable realism issues, but the overall intent is still understandable.\n"
                                "- **3:** Mostly plausible but with a few minor or localized physical issues.\n"
                                "- **4:** Physically realistic with only very small or hard‑to‑notice issues.\n"
                                "- **5:** Fully realistic: geometry, contact, gravity, and materials all look physically correct.\n\n"
                                "**Perceptual Quality (0–5)**\n"
                                "- **0–1:** Very broken or unreadable (heavy artifacts, extreme blur/noise, or severe temporal glitches in video).\n"
                                "- **2:** Many artifacts or inconsistencies, but you can still understand what is going on.\n"
                                "- **3:** Acceptable quality: generally clear, with some artifacts, blur, or jitter.\n"
                                "- **4:** High quality with only small, infrequent artifacts.\n"
                                "- **5:** Very clear and stable: sharp details, low noise, and (for video) consistent appearance across frames."
                            )

                    with gr.Tab("Comments"):
                        comments_box = gr.Textbox(
                            label="Additional Comments (optional)",
                            placeholder="Anything unclear, failure modes, or notable behaviors...",
                            lines=6,
                        )

                with gr.Row(elem_classes=["card"]):
                    prev_btn = gr.Button("⬅️ Previous", variant="secondary")
                    save_next_btn = gr.Button(
                        "Save & Next ➡️", variant="primary")

        # ---------- Callbacks ----------
        def ui_start(existing: str):
            (
                user_uuid,
                progress,
                state,
                rubric_rows,
                physical,
                perceptual,
                comments,
            ) = start_or_load_session(existing)

            instances = get_instances_for_uuid(user_uuid)
            instance = instances[state["current_index"]]

            # Media visibility
            if instance.media_type == "image":
                img_update = gr.update(value=instance.media_path, visible=True)
                vid_update = gr.update(value=None, visible=False)
            else:
                img_update = gr.update(value=None, visible=False)
                vid_update = gr.update(value=instance.media_path, visible=True)

            context = build_context_markdown(instance)

            # Rubric item boxes
            updates = []
            for i, (item_box, q_md, a_radio) in enumerate(rubric_rows_components):
                if i < len(rubric_rows):
                    label, answer = rubric_rows[i]
                    ans = (answer or "No").strip().title()
                    if ans not in ("Yes", "No", "Issue"):
                        ans = "No"
                    updates.extend([
                        gr.update(visible=True),          # item_box
                        gr.update(value=label),           # q_md
                        gr.update(value=ans),             # a_radio
                    ])
                else:
                    updates.extend([
                        gr.update(visible=False),
                        gr.update(value=""),
                        gr.update(value="No"),
                    ])

            phys_update = physical if physical is not None else 3
            perc_update = perceptual if perceptual is not None else 3
            comments_update = comments or ""

            return (
                progress,
                state,
                img_update,
                vid_update,
                context,
                *updates,
                phys_update,
                perc_update,
                comments_update,
            )

        start_btn.click(
            fn=ui_start,
            inputs=[existing_uuid],
            outputs=[
                progress_md,
                session_state,
                media_image,
                media_video,
                context_md,
                *[comp for triple in rubric_rows_components for comp in triple],
                physical_slider,
                perceptual_slider,
                comments_box,
            ],
        )

        def ui_prev(state: dict):
            if not state:
                # Reset UI cleanly
                empty_updates = []
                for _item_box, _q_md, _a_radio in rubric_rows_components:
                    empty_updates.extend([
                        gr.update(visible=False),
                        gr.update(value=""),
                        gr.update(value="No"),
                    ])
                return (
                    state,
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    "",
                    *empty_updates,
                    3,
                    3,
                    "",
                    "**Progress:** —",
                )

            state = go_previous(state)
            user_uuid = state.get("uuid")
            instances = get_instances_for_uuid(user_uuid)
            instance = instances[state["current_index"]]
            context, media_path, rubric_rows, physical, perceptual, progress = render_instance(
                state)

            if instance.media_type == "image":
                img_update = gr.update(value=media_path, visible=True)
                vid_update = gr.update(value=None, visible=False)
            else:
                img_update = gr.update(value=None, visible=False)
                vid_update = gr.update(value=media_path, visible=True)

            updates = []
            for i, (item_box, q_md, a_radio) in enumerate(rubric_rows_components):
                if i < len(rubric_rows):
                    label, answer = rubric_rows[i]
                    ans = (answer or "No").strip().title()
                    if ans not in ("Yes", "No", "Issue"):
                        ans = "No"
                    updates.extend([
                        gr.update(visible=True),
                        gr.update(value=label),
                        gr.update(value=ans),
                    ])
                else:
                    updates.extend([
                        gr.update(visible=False),
                        gr.update(value=""),
                        gr.update(value="No"),
                    ])

            phys_update = physical if physical is not None else 3
            perc_update = perceptual if perceptual is not None else 3

            return (
                state,
                img_update,
                vid_update,
                context,
                *updates,
                phys_update,
                perc_update,
                # keep comments empty on prev (or load from rec if you prefer)
                "",
                progress,
            )

        prev_btn.click(
            fn=ui_prev,
            inputs=[session_state],
            outputs=[
                session_state,
                media_image,
                media_video,
                context_md,
                *[comp for triple in rubric_rows_components for comp in triple],
                physical_slider,
                perceptual_slider,
                comments_box,
                progress_md,
            ],
        )

        def ui_save_next(state: dict, *vals):
            # vals: [radio0..radioN-1, physical, perceptual, comments]
            num_items = len(rubric_rows_components)
            if len(vals) < num_items + 3:
                radios = ["No"] * num_items
                physical = 3
                perceptual = 3
                comments = ""
            else:
                radios = list(vals[:num_items])
                physical = vals[num_items]
                perceptual = vals[num_items + 1]
                comments = vals[num_items + 2] or ""

            if not state:
                empty_updates = []
                for _item_box, _q_md, _a_radio in rubric_rows_components:
                    empty_updates.extend([
                        gr.update(visible=False),
                        gr.update(value=""),
                        gr.update(value="No"),
                    ])
                return (
                    state,
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                    "",
                    *empty_updates,
                    3,
                    3,
                    "",
                    "**Progress:** —",
                )

            user_uuid = state.get("uuid")
            instances = get_instances_for_uuid(user_uuid)
            idx = state.get("current_index", 0)
            instance = instances[idx]

            # Canonical label order
            tasks = get_tasks()
            base_rows, _ = build_rubric_table(tasks, instance, rec=None)
            labels = [row[0] for row in base_rows]

            rubric_table: List[List[str]] = []
            for i, label in enumerate(labels):
                raw = (radios[i] if i < len(radios) else "No") or "No"
                ans = str(raw).strip().title()
                if ans not in ("Yes", "No", "Issue"):
                    ans = "No"
                rubric_table.append([label, ans])

            state = save_and_advance(
                state, rubric_table, physical, perceptual, comments)

            # Render next
            context, media_path, rubric_rows, physical_v, perceptual_v, progress = render_instance(
                state)
            user_uuid = state.get("uuid")
            instances = get_instances_for_uuid(user_uuid)
            instance = instances[state["current_index"]]

            if instance.media_type == "image":
                img_update = gr.update(value=media_path, visible=True)
                vid_update = gr.update(value=None, visible=False)
            else:
                img_update = gr.update(value=None, visible=False)
                vid_update = gr.update(value=media_path, visible=True)

            updates = []
            for i, (item_box, q_md, a_radio) in enumerate(rubric_rows_components):
                if i < len(rubric_rows):
                    label, answer = rubric_rows[i]
                    ans = (answer or "No").strip().title()
                    if ans not in ("Yes", "No", "Issue"):
                        ans = "No"
                    updates.extend([
                        gr.update(visible=True),
                        gr.update(value=label),
                        gr.update(value=ans),
                    ])
                else:
                    updates.extend([
                        gr.update(visible=False),
                        gr.update(value=""),
                        gr.update(value="No"),
                    ])

            phys_update = physical_v if physical_v is not None else 3
            perc_update = perceptual_v if perceptual_v is not None else 3

            return (
                state,
                img_update,
                vid_update,
                context,
                *updates,
                phys_update,
                perc_update,
                # clear comment box after save (or keep it if you want)
                "",
                progress,
            )

        save_next_btn.click(
            fn=ui_save_next,
            inputs=[
                session_state,
                *[a_radio for (_item_box, _q_md, a_radio)
                  in rubric_rows_components],
                physical_slider,
                perceptual_slider,
                comments_box,
            ],
            outputs=[
                session_state,
                media_image,
                media_video,
                context_md,
                *[comp for triple in rubric_rows_components for comp in triple],
                physical_slider,
                perceptual_slider,
                comments_box,
                progress_md,
            ],
        )

    return demo


def main():
    app = build_app()
    app.launch(allowed_paths=[MEDIA_ROOT], share=True)


if __name__ == "__main__":
    main()
