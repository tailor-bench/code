import os
import sys
import json

sys.path.append(os.path.dirname(__file__))

from s1_task_generation import generate_task_from_action
from s2_creative_generation import generate_creative_solutions
from s3_adversarial_generation import generate_impossible_tools

def main():
    # Load action ontology
    ontology_path = os.path.join(os.path.dirname(__file__), "..", "action_ontology.json")
    with open(ontology_path, 'r') as f:
        actions = json.load(f)
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "tasks")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(actions)} actions...")
    
    # Step 1: Generate tasks from actions
    print("\n=== Step 1: Generating tasks from actions ===")
    for action in actions:
        action_name = action["action_name"]
        print(f"Processing: {action_name}")
        task_data = generate_task_from_action(action, output_dir)
        if task_data:
            count = len(task_data.get('tasks', []))
            print(f"  ✓ Generated {count} tasks")
        else:
            print(f"  ✗ Failed to generate task")
    
    # Step 2: Generate creative solutions
    print("\n=== Step 2: Generating creative solutions ===")
    for action in actions:
        action_name = action["action_name"]
        s1_file = os.path.join(output_dir, action_name, "s1_task.json")
        
        if os.path.exists(s1_file):
            print(f"Processing: {action_name}")
            with open(s1_file, 'r') as f:
                s1_data = json.load(f)
            tasks = s1_data.get("tasks", [s1_data] if "task_goal" in s1_data else [])
            results = []
            for i, task_data in enumerate(tasks):
                result = generate_creative_solutions(task_data, output_dir, action_name, save=False)
                if result:
                    results.append(result)
            if results:
                s2_file = os.path.join(output_dir, action_name, "s2_creative.json")
                with open(s2_file, 'w') as f:
                    json.dump({"tasks": results}, f, indent=4)
                print(f"  ✓ Generated creative solutions for {len(results)} tasks")
            else:
                print(f"  ✗ Failed to generate creative solutions")
        else:
            print(f"Skipping {action_name}: s1_task.json not found")
    
    # Step 3: Generate impossible tools
    print("\n=== Step 3: Generating impossible tools ===")
    for action in actions:
        action_name = action["action_name"]
        s2_file = os.path.join(output_dir, action_name, "s2_creative.json")
        
        if os.path.exists(s2_file):
            print(f"Processing: {action_name}")
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
                print(f"  ✓ Generated impossible tools for {len(results)} tasks")
                print(f"  ✓ Final tasks saved to: tasks/{action_name}/final_task.json")
            else:
                print(f"  ✗ Failed to generate impossible tools")
        else:
            print(f"Skipping {action_name}: s2_creative.json not found")
    
    print("\n=== All steps completed! ===")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
