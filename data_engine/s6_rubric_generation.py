import json
import os
import re
import sys
from typing import Any, Dict, List

sys.path.append(os.path.dirname(__file__))

from generate import generate  # type: ignore[import]
from prompts.s6_rubric_generation import PROMPT  # type: ignore[import]


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def generate_rubric_for_evaluation_instance(
    task: Dict[str, Any], eval_instance: Dict[str, Any]
) -> Dict[str, Any]:
    """Call the LLM once for this evaluation instance to generate rubrics for
    all four prompts (predictive/description × image/video)."""
    task_goal = task.get("task_goal", "")
    action_type = task.get("action_type", "")
    original_tool = task.get("original_tool", "")
    expected_outcome = task.get("expected_outcome", "")

    required_tool_attributes = _ensure_list(task.get("required_tool_attributes"))
    unconventional_tools = _ensure_list(task.get("unconventional_tools"))
    impossible_tools = _ensure_list(task.get("impossible_tools"))

    tool_type = eval_instance.get("tool_type", "")
    tool = eval_instance.get("tool", "")
    instance_expected_outcome = eval_instance.get("expected_outcome", "")
    predictive_image_prompt = eval_instance.get("predictive_image_prompt", "")
    descriptive_image_prompt = eval_instance.get("descriptive_image_prompt", "")
    predictive_video_prompt = eval_instance.get("predictive_video_prompt", "")
    descriptive_video_prompt = eval_instance.get("descriptive_video_prompt", "")

    prompt = PROMPT.format(
        task_goal=task_goal,
        action_type=action_type,
        original_tool=original_tool,
        expected_outcome=expected_outcome,
        required_tool_attributes=json.dumps(
            required_tool_attributes, indent=2, ensure_ascii=False
        ),
        unconventional_tools=json.dumps(
            unconventional_tools, indent=2, ensure_ascii=False
        ),
        impossible_tools=json.dumps(
            impossible_tools, indent=2, ensure_ascii=False
        ),
        tool_type=tool_type,
        tool=tool,
        instance_expected_outcome=instance_expected_outcome,
        predictive_image_prompt=predictive_image_prompt,
        descriptive_image_prompt=descriptive_image_prompt,
        predictive_video_prompt=predictive_video_prompt,
        descriptive_video_prompt=descriptive_video_prompt,
    )

    response, _ = generate(prompt)

    augmented_instance = dict(eval_instance)
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(response)
    except json.JSONDecodeError as e:
        print(
            f"Error parsing rubric JSON for task '{task_goal}' and tool '{tool}': {e}"
        )
        print(f"Response was:\n{response}\n")
        return augmented_instance

    # Attach up to four per-prompt rubrics if present.
    for key in [
        "predictive_image_rubric",
        "descriptive_image_rubric",
        "predictive_video_rubric",
        "descriptive_video_rubric",
    ]:
        rubric_obj = parsed.get(key)
        if rubric_obj is not None:
            augmented_instance[key] = rubric_obj
    return augmented_instance


def generate_rubrics_for_file(
    input_path: str, output_path: str
) -> List[Dict[str, Any]]:
    """Load tasks with evaluation instances, call LLM to generate rubrics, and save."""
    with open(input_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    else:
        tasks = data

    if not isinstance(tasks, list):
        raise ValueError("Expected a list of task objects or a dict with 'tasks' key.")

    augmented_tasks: List[Dict[str, Any]] = []
    total_instances = 0

    for task in tasks:
        eval_instances = task.get("evaluation_instances", [])
        new_eval_instances = [
            generate_rubric_for_evaluation_instance(task, ei) for ei in eval_instances
        ]
        new_task = dict(task)
        new_task["evaluation_instances"] = new_eval_instances
        augmented_tasks.append(new_task)
        total_instances += len(new_eval_instances)

    if isinstance(data, dict) and "tasks" in data:
        out_data: Any = dict(data)
        out_data["tasks"] = augmented_tasks
    else:
        out_data = augmented_tasks

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=4, ensure_ascii=False)

    print(
        f"Generated rubrics for {total_instances} evaluation instances "
        f"across {len(augmented_tasks)} tasks."
    )
    print(f"Saved augmented tasks with rubrics to: {output_path}")

    return augmented_tasks


def main() -> None:
    base_dir = os.path.dirname(__file__)
    tasks_dir = os.path.join(base_dir, "..", "tasks")

    input_path = os.path.join(tasks_dir, "selected_tasks_with_prompts.json")
    output_path = os.path.join(tasks_dir, "selected_tasks_with_rubrics.json")

    print(f"Loading tasks with prompts from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run s5_prompt_generation.py first to create it."
        )

    generate_rubrics_for_file(input_path, output_path)


if __name__ == "__main__":
    main()

