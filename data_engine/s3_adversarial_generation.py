"""Step 3: Generate opposite attributes and impossible tools"""
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
from generate import generate
from prompts.s3_task_impossible_solution import PROMPT

def generate_impossible_tools(task_data, output_dir, action_name, save=True):
    """Generate opposite attributes and impossible tools"""
    prompt = PROMPT.format(
        task_goal=task_data["task_goal"],
        original_tool=task_data["original_tool"],
        required_tool_attributes=json.dumps(task_data["required_tool_attributes"], indent=2)
    )
    
    response, _ = generate(prompt)
    
    # Parse JSON response
    try:
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            impossible_data = json.loads(json_match.group())
        else:
            impossible_data = json.loads(response)
        
        # Merge with existing task data
        task_data.update(impossible_data)
        
        # Save final task data (when not in batch mode)
        if save:
            output_file = os.path.join(output_dir, action_name, "final_task.json")
            with open(output_file, 'w') as f:
                json.dump(task_data, f, indent=4)
        
        return task_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for {action_name}: {e}")
        print(f"Response was: {response}")
        return None

if __name__ == "__main__":
    # Load action ontology
    ontology_path = os.path.join(os.path.dirname(__file__), "..", "action_ontology.json")
    with open(ontology_path, 'r') as f:
        actions = json.load(f)
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "tasks")
    
    # Process each action
    for action in actions:
        action_name = action["action_name"]
        s2_file = os.path.join(output_dir, action_name, "s2_creative.json")
        
        if os.path.exists(s2_file):
            print(f"Generating impossible tools for: {action_name}")
            with open(s2_file, 'r') as f:
                s2_data = json.load(f)
            tasks = s2_data.get("tasks", [s2_data] if "task_goal" in s2_data else [])
            results = []
            for task_data in tasks:
                result = generate_impossible_tools(task_data, output_dir, action_name, save=False)
                if result:
                    results.append(result)
            if results:
                output_file = os.path.join(output_dir, action_name, "final_task.json")
                with open(output_file, 'w') as f:
                    json.dump({"tasks": results}, f, indent=4)
                print(f"  Processed {len(results)} tasks")
        else:
            print(f"Skipping {action_name}: s2_creative.json not found")
