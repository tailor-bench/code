import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.dirname(__file__))

from s4_revise_instances import (  # type: ignore[import-untyped]
    FILTERED_PATH as S4_FILTERED_PATH,
    load_filtered_tasks,
    refine_task,
)
from s5_prompt_generation import generate_evaluation_instances_for_task  # type: ignore[import-untyped]
from s6_rubric_generation import generate_rubric_for_evaluation_instance  # type: ignore[import-untyped]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(SCRIPT_DIR, "..", "tasks")
REVISED_PATH = os.path.join(TASKS_DIR, "revised_instances.json")
PROMPTS_PATH = os.path.join(TASKS_DIR, "all_tasks_with_prompts.json")
RUBRICS_PATH = os.path.join(TASKS_DIR, "all_tasks_with_rubrics.json")


def _run_s4_parallel(
    tasks: List[Dict[str, Any]],
    output_path: str,
    jobs: int,
) -> List[Dict[str, Any]]:
    """Revise tasks in parallel; preserve order and fallback on parse error."""
    results: List[Tuple[int, Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_idx = {executor.submit(refine_task, task): i for i, task in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                refined, err, raw = future.result()
                if refined is not None:
                    results.append((i, refined))
                else:
                    task = tasks[i]
                    fallback = dict(task)
                    fallback.pop("expected_outcome_impossible_tool", None)
                    results.append((i, fallback))
                    print(f"  [{i+1}] JSONDecodeError: {err}")
            except Exception as e:
                task = tasks[i]
                fallback = dict(task)
                fallback.pop("expected_outcome_impossible_tool", None)
                results.append((i, fallback))
                print(f"  [{i+1}] Error: {e}")

    ordered = [r[1] for r in sorted(results, key=lambda x: x[0])]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ordered, f, indent=2)
    return ordered


def _run_s5_parallel(
    tasks: List[Dict[str, Any]],
    output_path: str,
    jobs: int,
) -> List[Dict[str, Any]]:
    """Generate evaluation instances per task in parallel; preserve order."""
    results: List[Tuple[int, Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_idx = {
            executor.submit(generate_evaluation_instances_for_task, task): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                augmented = future.result()
                results.append((i, augmented))
            except Exception as e:
                print(f"  Task {i+1} failed: {e}")
                results.append((i, tasks[i]))

    ordered = [r[1] for r in sorted(results, key=lambda x: x[0])]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ordered, f, indent=4, ensure_ascii=False)
    return ordered


def _run_s6_parallel(
    tasks: List[Dict[str, Any]],
    output_path: str,
    jobs: int,
) -> List[Dict[str, Any]]:
    """Generate rubrics per evaluation instance in parallel; preserve order."""
    # Flatten to (task_idx, eval_idx, task, eval_instance)
    work: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]] = []
    for ti, task in enumerate(tasks):
        for ei, eval_instance in enumerate(task.get("evaluation_instances", [])):
            work.append((ti, ei, task, eval_instance))

    def do_one(item: Tuple[int, int, Dict[str, Any], Dict[str, Any]]) -> Tuple[int, int, Dict[str, Any]]:
        ti, ei, task, eval_instance = item
        augmented = generate_rubric_for_evaluation_instance(task, eval_instance)
        return (ti, ei, augmented)

    instance_results: List[Tuple[int, int, Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_item = {executor.submit(do_one, item): item for item in work}
        for future in as_completed(future_to_item):
            try:
                instance_results.append(future.result())
            except Exception as e:
                ti, ei, task, eval_instance = future_to_item[future]
                print(f"  Task {ti} instance {ei} failed: {e}")
                instance_results.append((ti, ei, dict(eval_instance)))

    # Group by task_idx, sort eval_idx per task
    by_task: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for ti, ei, aug in instance_results:
        by_task[ti].append((ei, aug))
    augmented_tasks = []
    for ti in range(len(tasks)):
        instances = sorted(by_task[ti], key=lambda x: x[0])
        new_task = dict(tasks[ti])
        new_task["evaluation_instances"] = [inst[1] for inst in instances]
        augmented_tasks.append(new_task)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(augmented_tasks, f, indent=4, ensure_ascii=False)
    return augmented_tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stage 2 (s4 → s5 → s6) with parallel LLM calls."
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Max number of tasks for s4 (for testing)",
    )
    parser.add_argument(
        "--filtered", default=None,
        help="Path to filtered_tasks.json (default: ../tasks/filtered_tasks.json)",
    )
    parser.add_argument(
        "--revised", default=None,
        help="Output path for s4 revised_instances.json",
    )
    parser.add_argument(
        "--prompts", default=None,
        help="Output path for s5 all_tasks_with_prompts.json",
    )
    parser.add_argument(
        "--rubrics", default=None,
        help="Output path for s6 all_tasks_with_rubrics.json",
    )
    args = parser.parse_args()

    filtered_path = args.filtered or S4_FILTERED_PATH
    revised_path = args.revised or REVISED_PATH
    prompts_path = args.prompts or PROMPTS_PATH
    rubrics_path = args.rubrics or RUBRICS_PATH
    jobs = max(1, args.jobs)

    print("=== Stage 2 (parallel): s4 → s5 → s6 ===\n")

    # Step 4
    print(f"--- Step 4: Revise eval instances ({jobs} workers) ---")
    tasks = load_filtered_tasks(filtered_path)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    print(f"Processing {len(tasks)} tasks...")
    revised = _run_s4_parallel(tasks, revised_path, jobs)
    print(f"Wrote {len(revised)} tasks to {revised_path}")

    # Step 5
    print(f"\n--- Step 5: Prompt generation ({jobs} workers) ---")
    print(f"Loading from {revised_path}...")
    tasks_with_prompts = _run_s5_parallel(revised, prompts_path, jobs)
    total_s5 = sum(len(t.get("evaluation_instances", [])) for t in tasks_with_prompts)
    print(f"Generated {total_s5} evaluation instances across {len(tasks_with_prompts)} tasks.")
    print(f"Saved to {prompts_path}")

    # Step 6
    print(f"\n--- Step 6: Rubric generation ({jobs} workers) ---")
    total_s6 = sum(len(t.get("evaluation_instances", [])) for t in tasks_with_prompts)
    print(f"Processing {total_s6} evaluation instances...")
    _run_s6_parallel(tasks_with_prompts, rubrics_path, jobs)
    print(f"Saved to {rubrics_path}")

    print("\n=== Stage 2 (parallel) complete ===")


if __name__ == "__main__":
    main()
