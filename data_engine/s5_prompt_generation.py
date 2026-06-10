import json
import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.dirname(__file__))

from generate import generate  # type: ignore[import]
from prompts.s5_prompt_generation import PROMPT  # type: ignore[import]


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def generate_evaluation_instances_for_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM once for this task to generate 5 evaluation instances."""
    task_goal = task.get("task_goal", "")
    original_tool = task.get("original_tool", "")
    expected_outcome = task.get("expected_outcome", "")

    unconventional_tools = _ensure_list(task.get("unconventional_tools"))
    impossible_tools = _ensure_list(task.get("impossible_tools"))

    prompt = PROMPT.format(
        task_goal=task_goal,
        original_tool=original_tool,
        expected_outcome=expected_outcome,
        unconventional_tools=json.dumps(unconventional_tools, indent=2, ensure_ascii=False),
        impossible_tools=json.dumps(impossible_tools, indent=2, ensure_ascii=False),
    )

    response, _ = generate(prompt)

    try:
        import re

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for task '{task_goal}': {e}")
        print(f"Response was:\n{response}\n")
        return task

    eval_instances = parsed.get("evaluation_instances", [])
    augmented = dict(task)
    augmented["evaluation_instances"] = eval_instances
    return augmented


def generate_prompts_for_file(input_path: str, output_path: str) -> List[Dict[str, Any]]:
    """Load tasks from input_path, call LLM to generate prompts, and save to output_path."""
    with open(input_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    else:
        tasks = data

    if not isinstance(tasks, list):
        raise ValueError("Expected a list of task objects or a dict with 'tasks' key.")

    augmented_tasks = [generate_evaluation_instances_for_task(task) for task in tasks]

    if isinstance(data, dict) and "tasks" in data:
        out_data: Any = dict(data)
        out_data["tasks"] = augmented_tasks
    else:
        out_data = augmented_tasks

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=4, ensure_ascii=False)

    return augmented_tasks


def main() -> None:
    base_dir = os.path.dirname(__file__)
    tasks_dir = os.path.join(base_dir, "..", "tasks")

    input_path = os.path.join(tasks_dir, "revised_instances.json")
    prompts_output_path = os.path.join(tasks_dir, "selected_tasks_with_prompts.json")

    print(f"Loading tasks from: {input_path}")
    tasks_with_prompts = generate_prompts_for_file(input_path, prompts_output_path)
    total_instances = sum(len(t.get("evaluation_instances", [])) for t in tasks_with_prompts)
    print(f"Generated {total_instances} evaluation instances across {len(tasks_with_prompts)} tasks.")
    print(f"Saved augmented tasks with prompts to: {prompts_output_path}")

if __name__ == "__main__":
    main()


