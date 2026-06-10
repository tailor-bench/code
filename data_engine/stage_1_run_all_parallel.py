import os
import sys
import json
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.append(os.path.dirname(__file__))

from s1_task_generation import generate_task_from_action
from s2_creative_generation import generate_creative_solutions
from s3_adversarial_generation import generate_impossible_tools

def process_step1(action, output_dir):
    """Process a single action for step 1"""
    action_name = action["action_name"]
    try:
        task_data = generate_task_from_action(action, output_dir)
        if task_data:
            count = len(task_data.get('tasks', []))
            return (action_name, True, count, None)
        else:
            return (action_name, False, 0, "Failed to generate task")
    except Exception as e:
        return (action_name, False, 0, str(e))

def process_step2(action, output_dir):
    """Process a single action for step 2"""
    action_name = action["action_name"]
    try:
        s1_file = os.path.join(output_dir, action_name, "s1_task.json")
        
        if not os.path.exists(s1_file):
            return (action_name, False, 0, "s1_task.json not found")
        
        with open(s1_file, 'r') as f:
            s1_data = json.load(f)
        tasks = s1_data.get("tasks", [s1_data] if "task_goal" in s1_data else [])
        results = []
        
        for task_data in tasks:
            result = generate_creative_solutions(task_data, output_dir, action_name, save=False)
            if result:
                results.append(result)
        
        if results:
            s2_file = os.path.join(output_dir, action_name, "s2_creative.json")
            with open(s2_file, 'w') as f:
                json.dump({"tasks": results}, f, indent=4)
            return (action_name, True, len(results), None)
        else:
            return (action_name, False, 0, "Failed to generate creative solutions")
    except Exception as e:
        return (action_name, False, 0, str(e))

def process_step3(action, output_dir):
    """Process a single action for step 3"""
    action_name = action["action_name"]
    try:
        s2_file = os.path.join(output_dir, action_name, "s2_creative.json")
        
        if not os.path.exists(s2_file):
            return (action_name, False, 0, "s2_creative.json not found")
        
        with open(s2_file, 'r') as f:
            s2_data = json.load(f)
        tasks = s2_data.get("tasks", [s2_data] if "task_goal" in s2_data else [])
        results = []
        
        for task_data in tasks:
            result = generate_impossible_tools(task_data, output_dir, action_name, save=False)
            if result:
                results.append(result)
        
        if results:
            final_file = os.path.join(output_dir, action_name, "final_task.json")
            with open(final_file, 'w') as f:
                json.dump({"tasks": results}, f, indent=4)
            return (action_name, True, len(results), None)
        else:
            return (action_name, False, 0, "Failed to generate impossible tools")
    except Exception as e:
        return (action_name, False, 0, str(e))

def main(num_workers=None):
    # Load action ontology
    ontology_path = os.path.join(os.path.dirname(__file__), "..", "action_ontology.json")
    with open(ontology_path, 'r') as f:
        actions = json.load(f)
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "tasks")
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(actions))
    
    print(f"Processing {len(actions)} actions with {num_workers} workers...")
    
    # Step 1: Generate tasks from actions
    print("\n=== Step 1: Generating tasks from actions ===")
    with Pool(processes=num_workers) as pool:
        process_func = partial(process_step1, output_dir=output_dir)
        results = pool.map(process_func, actions)
    
    for action_name, success, count, error in results:
        if success:
            print(f"  ✓ {action_name}: Generated {count} tasks")
        else:
            print(f"  ✗ {action_name}: {error if error else 'Failed'}")
    
    # Step 2: Generate creative solutions
    print("\n=== Step 2: Generating creative solutions ===")
    with Pool(processes=num_workers) as pool:
        process_func = partial(process_step2, output_dir=output_dir)
        results = pool.map(process_func, actions)
    
    for action_name, success, count, error in results:
        if success:
            print(f"  ✓ {action_name}: Generated creative solutions for {count} tasks")
        elif "not found" in str(error):
            print(f"  ⊘ {action_name}: Skipped ({error})")
        else:
            print(f"  ✗ {action_name}: {error if error else 'Failed'}")
    
    # Step 3: Generate impossible tools
    print("\n=== Step 3: Generating impossible tools ===")
    with Pool(processes=num_workers) as pool:
        process_func = partial(process_step3, output_dir=output_dir)
        results = pool.map(process_func, actions)
    
    for action_name, success, count, error in results:
        if success:
            print(f"  ✓ {action_name}: Generated impossible tools for {count} tasks")
        elif "not found" in str(error):
            print(f"  ⊘ {action_name}: Skipped ({error})")
        else:
            print(f"  ✗ {action_name}: {error if error else 'Failed'}")
    
    print("\n=== All steps completed! ===")
    print(f"Results saved in: {output_dir}")

    # save all tasks to a single file
    all_tasks = {}
    for action_name in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, action_name)):
            final_file = os.path.join(output_dir, action_name, "final_task.json")
            if os.path.exists(final_file):
                with open(final_file, 'r') as f:
                    final_data = json.load(f)
                    final_tasks = final_data.get("tasks", [])
                    if len(final_tasks) > 0:
                        all_tasks[action_name] = final_tasks
    with open(os.path.join(output_dir, "all_tasks.json"), 'w') as f:
        json.dump(all_tasks, f, indent=4)
    print(f"All tasks saved to: {os.path.join(output_dir, 'all_tasks.json')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run task generation pipeline in parallel")
    parser.add_argument("-j", "--jobs", type=int, default=10,
                        help="Number of parallel workers (default: number of CPU cores)")
    args = parser.parse_args()
    main(num_workers=args.jobs)
