"""Step 4: Revise eval instances for visual generation models (one LLM call per task)."""
import argparse
import json
import os
import re
import sys

sys.path.append(os.path.dirname(__file__))
from generate import generate
from prompts.s4_revise_eval_instances import PROMPT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(SCRIPT_DIR, "..", "tasks")
FILTERED_PATH = os.path.join(TASKS_DIR, "filtered_tasks.json")
DEFAULT_OUTPUT_PATH = os.path.join(TASKS_DIR, "revised_instances.json")

REQUIRED_KEYS = [
    "action_type",
    "task_goal",
    "original_tool",
    "expected_outcome",
    "required_tool_attributes",
    "opposite_tool_attributes",
    "unconventional_tools",
    "impossible_tools",
]


def load_filtered_tasks(path):
    """Load filtered_tasks.json and flatten to list of tasks with action_type."""
    with open(path, "r") as f:
        data = json.load(f)
    tasks = []
    for action_type, task_list in data.items():
        for t in task_list:
            task = dict(t)
            task["action_type"] = action_type
            tasks.append(task)
    return tasks


def parse_json_response(response):
    """Extract JSON object from model response (strip markdown/code blocks if needed)."""
    text = response.strip()
    # Remove optional markdown code fence
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def refine_task(task, prompt_template=PROMPT):
    """Run one LLM call to refine the task; return refined task or None on parse error."""
    task_json = json.dumps(task, indent=2)
    prompt = prompt_template.replace("{task_json}", task_json)
    response, _ = generate(prompt)
    try:
        refined = parse_json_response(response)
    except json.JSONDecodeError as e:
        return None, str(e), response
    # Ensure action_type present
    if "action_type" not in refined and "action_type" in task:
        refined["action_type"] = task["action_type"]
    for key in REQUIRED_KEYS:
        if key not in refined and key in task:
            refined[key] = task[key]
    refined.pop("expected_outcome_impossible_tool", None)
    return refined, None, None


def main():
    parser = argparse.ArgumentParser(description="Revise eval instances with LLM (one call per task).")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_PATH, help="Output JSON path")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Max number of tasks to process (for testing)")
    parser.add_argument("--filtered", default=FILTERED_PATH, help="Path to filtered_tasks.json")
    args = parser.parse_args()

    tasks = load_filtered_tasks(args.filtered)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    results = []
    for i, task in enumerate(tasks):
        task_id = task.get("task_goal", str(i))
        print(f"[{i + 1}/{len(tasks)}] {task_id[:60]}...")
        refined, err, raw = refine_task(task)
        if refined is not None:
            results.append(refined)
        else:
            print(f"  JSONDecodeError: {err}")
            print(f"  Raw response (first 500 chars): {raw[:500] if raw else ''}")
            fallback = dict(task)
            fallback.pop("expected_outcome_impossible_tool", None)
            results.append(fallback)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} tasks to {args.output}")


if __name__ == "__main__":
    main()
